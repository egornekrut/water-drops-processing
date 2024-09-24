from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from albumentations import Compose, PadIfNeeded
from albumentations.pytorch import ToTensorV2
from cv2 import BORDER_CONSTANT
from easydict import EasyDict
from PIL import Image, ImageDraw
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base import SegmentationModel

from src.utils.inference_v2 import BasicModelPipeline
from src.utils.masks_and_bboxes import determine_bboxes, get_bbox_from_mask, xywh_xyxy


class BubbleSegmentation(BasicModelPipeline):
    def __init__(
            self,
            model_config: EasyDict,
            device: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_config,
            ['image', 'full_mask_bool'],
            ['bubbles_mask_raw', 'bubbles_mask'],
            device,
        )
        self.divider = 32
        self.transforms = Compose([
            PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                border_mode=BORDER_CONSTANT,
                position=PadIfNeeded.PositionType.TOP_LEFT,
                value=127,
            ),
            ToTensorV2(),
        ])

    def _setup_model(self) -> SegmentationModel:
        model = Unet(
            encoder_name=self.model_config.encoder_name,
            encoder_weights=None,
            in_channels=1,
            classes=1,
            decoder_attention_type='scse',
            activation='sigmoid',
        )

        state_dict = torch.load(self.model_config.weights, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def _preprocess(self, input_data: Dict[str,  Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        full_mask = input_data['full_mask_bool'][..., 0]
        if full_mask.sum() == 0:
            return torch.zeros((1, 1, 32, 32), dtype=torch.float32, device=self.device)

        bbox_coords = get_bbox_from_mask(full_mask)
        crop_np = np.asarray(input_data['image'].convert('L').crop(bbox_coords))

        transformed: torch.Tensor = self.transforms(image=crop_np)['image']
        img_tensor = transformed.to(dtype=torch.float32, device=self.device).unsqueeze(0) / 127.5 - 1
        return img_tensor

    def _postprocess(self, model_mask: torch.Tensor) -> Dict[str, Any]:
        """Postprocess model output before returning it to the next pipeline.

        Args:
            model_output (Any): Model output

        Returns:
            Any: Postprocessed data
        """
        mask_raw = model_mask.squeeze((0, 1)).cpu().numpy()
        mask_result = (mask_raw > self.model_config.thres).astype('uint8') * 255
        bubble_boxes = determine_bboxes(mask_result)
        stat = {
            'bubbles_area': np.count_nonzero(mask_result),
            'num_bubbles': len(bubble_boxes),
        }

        if mask_result.sum() == 0:
            mask_result = None
        else:
            mask_result = Image.fromarray(mask_result).convert('RGB')
            mask_result_draw = ImageDraw.Draw(mask_result)
            img_size = mask_result.size
            for box in bubble_boxes:
                bbox_coords = xywh_xyxy(box, img_size)
                mask_result_draw.rectangle(bbox_coords, outline='red')

        return {'bubbles_mask_raw': mask_raw, 'bubbles_mask': mask_result, 'bubbles_stat': stat}
