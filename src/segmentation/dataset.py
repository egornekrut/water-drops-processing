from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from albumentations import Compose, Flip, PadIfNeeded, RandomCrop, RandomBrightnessContrast, Rotate, Affine
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class BubDataset(Dataset):
    def __init__(
            self,
            config,
            inference_mode: bool = False,
        ):
        self.config = config

        dataset_root: Path = self.config.dataset_root if not inference_mode else self.config.test_root
        original_paths = list((dataset_root / 'original').glob('*.png'))

        self.images_dict = {}
        self.gt_dict = {}
        self.idx_mapping = {}

        for index, path in enumerate(original_paths):
            name = path.stem
            name_wo_orig = name.split('_original')[0]
            self.images_dict[name_wo_orig] = path
            gt_path = dataset_root / 'mask' / f'{name_wo_orig}_mask.png'
            if not gt_path.exists():
                raise FileNotFoundError(f'{gt_path} is not exists in dataset folder!')
            self.gt_dict[name_wo_orig] = gt_path
            self.idx_mapping[index] = name_wo_orig

        if inference_mode:
            self.transforms = Compose(
                [
                    PadIfNeeded(
                        min_height=None,
                        min_width=None,
                        pad_height_divisor=32,
                        pad_width_divisor=32,
                    ),
                    ToTensorV2(),
                ],
            )
        else:
            self.transforms = Compose(
                [
                    # PadIfNeeded(
                    #     min_height=None,
                    #     min_width=None,
                    #     pad_height_divisor=32,
                    #     pad_width_divisor=32,
                    # ),
                    # Rotate(limit=60, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=127, mask_value=0),
                    # Affine(),
                    RandomCrop(
                        height=480,
                        width=480,
                    ),
                    Flip(),
                    RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.3),
                    ToTensorV2(),
                ],
            )

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        name = self.idx_mapping[index]
        img = np.asarray(Image.open(self.images_dict[name]).convert('L')).copy()
        gt = np.asarray(Image.open(self.gt_dict[name]).convert('L')).copy()

        transformed = self.transforms(image=img, mask=gt)

        img_tensor = transformed['image'].to(torch.float32) / 127.5 - 1
        gt_tensor = torch.unsqueeze((transformed['mask'] > 0).to(torch.float32), dim=0)

        return img_tensor, gt_tensor

    def __len__(self):
        return len(self.idx_mapping)


class InferenceDataset:
    def __init__(self):
        self.transforms = Compose([
            PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                border_mode=cv2.BORDER_CONSTANT,
                position=PadIfNeeded.PositionType.TOP_LEFT,
            ),
            ToTensorV2(),
        ])

    def __call__(self, input_data: Image, **kwargs) -> torch.Tensor:
        in_data = np.asarray(input_data.convert('L'))
        padded = self.transforms(image=in_data)['image'] / 127.5 - 1

        return padded

    @staticmethod
    def decode(tensor: torch.Tensor) -> Image:
        tensor_denorm = torch.clip((tensor[0] + 1) * 127.5, min=0., max=255.).to(torch.uint8)
        pil_image = Image.fromarray(tensor_denorm.numpy())

        return pil_image
