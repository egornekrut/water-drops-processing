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
            ['bubbles_mask_raw', 'bubbles_mask', 'bubbles_stat', 'original_bubble_masked'],
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

    @torch.inference_mode()
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        input_data.update(self._preprocess(input_data))
        input_data.update({'model_result': self._model_inference(input_data['input_tensor'])})

        return self._postprocess(input_data)

    def _preprocess(self, input_data: Dict[str,  Union[np.ndarray, Image.Image]]) -> Dict[str, Any]:
        full_mask = input_data['full_mask_bool'][..., 0]
        if full_mask.sum() == 0:
            return {'input_tensor': torch.zeros((1, 1, 32, 32), dtype=torch.float32, device=self.device), 'image_crop_for_seg': None}

        bbox_coords = get_bbox_from_mask(full_mask)

        crop_pil = input_data['image'].convert('L').crop(bbox_coords)
        crop_np = np.asarray(crop_pil)

        transformed: torch.Tensor = self.transforms(image=crop_np)['image']
        img_tensor = transformed.to(dtype=torch.float32, device=self.device).unsqueeze(0) / 127.5 - 1

        return {'input_tensor': img_tensor, 'image_crop_for_seg': crop_pil}

    def _postprocess(self, model_result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess model output before returning it to the next pipeline.

        Args:
            model_output (Any): Model output

        Returns:
            Any: Postprocessed data
        """
        mask_raw = model_result['model_result'].squeeze((0, 1)).cpu().numpy()
        mask_result = (mask_raw > self.model_config.thres).astype('uint8') * 255
        bubble_boxes = determine_bboxes(mask_result)
        total_diams = {}
        for bbox in bubble_boxes:
            diam = int(np.mean((bbox[2] - bbox[0], bbox[3] - bbox[1])))
            if diam in total_diams:
                total_diams[diam] += 1
            else:
                total_diams[diam] = 1

        stat = {
            'bubbles_area': np.count_nonzero(mask_result),
            'num_bubbles': len(bubble_boxes),
            'diam_hist': total_diams,
        }

        if mask_result.sum() == 0:
            mask_result = None
            image_crop_for_seg = None
        else:
            mask_result = Image.fromarray(mask_result).convert('RGB')
            image_crop_for_seg = model_result['image_crop_for_seg'].convert('RGB')

            mask_result_draw = ImageDraw.Draw(mask_result)
            image_crop_for_seg_draw = ImageDraw.Draw(image_crop_for_seg)

            img_size = mask_result.size
            for box in bubble_boxes:
                # TODO: Нормально переименовать переменные и убрать перевороты в координатах
                bbox_coords = xywh_xyxy(box, img_size[::-1])
                # bbox_coords = [bbox_coords[1], bbox_coords[0], bbox_coords[3], bbox_coords[2]]
                mask_result_draw.rectangle(bbox_coords, outline='red')
                image_crop_for_seg_draw.rectangle(bbox_coords, outline='red')

        return {'bubbles_mask_raw': mask_raw, 'bubbles_mask': mask_result, 'bubbles_stat': stat, 'original_bubble_masked': image_crop_for_seg}
