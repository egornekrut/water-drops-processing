from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from albumentations import Compose
from albumentations import Flip
from albumentations import PadIfNeeded
from albumentations import RandomCrop
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class BubDataset(Dataset):
    def __init__(self, config):
        self.config = config

        dataset_root: Path = self.config.dataset_root
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

        self.transforms = Compose(
            [
                PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=32,
                    pad_width_divisor=32,
                ),
                # RandomCrop(64, 64),
                Flip(),
                ToTensorV2(),
            ],
        )

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        name = self.idx_mapping[index]
        img = np.asarray(Image.open(self.images_dict[name]).convert('L'))
        gt = np.asarray(Image.open(self.gt_dict[name]).convert('L'))

        transformed = self.transforms(image=img, mask=gt)

        img_tensor = transformed['image'].to(torch.float32) / 127.5 - 1
        gt_tensor = (transformed['mask'] > 0).to(torch.float32)

        return img_tensor, torch.unsqueeze(gt_tensor, dim=0)

    def __len__(self):
        return len(self.idx_mapping)


class InferenceProcessor:
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
