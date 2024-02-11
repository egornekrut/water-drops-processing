from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from albumentations import (Compose, GaussNoise, RandomBrightnessContrast,
                            ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset
import pims

class FFDataset(Dataset):
    def __init__(self, data_root: Path = Path('/mnt/c/Users/egorn/Desktop/WDP/fframe_dataset/normalized_npy')) -> None:
        super().__init__()
        self.data_root = data_root

        self.dataset = {}
        self.has_contact = {}

        npy_files = list(self.data_root.glob('*.npy'))
        self.train_label = np.empty((len(npy_files), 3), dtype=bool)

        for idx, npy_path in enumerate(npy_files):
            self.dataset[idx] = npy_path
            name_parts = npy_path.stem.split('_')
            npy_has_frames = [int(frame) for frame in name_parts[1].split('-')]
            npy_has_frames_arr = np.array((npy_has_frames[0], npy_has_frames[1] - 1, npy_has_frames[1])) == int(name_parts[2]) + 1
            self.train_label[idx, :] = npy_has_frames_arr
            self.has_contact[idx] = int(npy_has_frames_arr.max())

        self.augs = Compose(
            [
                RandomBrightnessContrast(0.01, 0.01, p=0.5),
                GaussNoise(var_limit=(0.0001, 0.0005), p=0.5),
                ShiftScaleRotate(border_mode=0),
            ]
        )
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # d, w, h --> w, h, d
        image_series = np.transpose(np.load(self.dataset[index]), (1, 2, 0))
        image_series = self.augs(image=image_series)['image']
        # w, h, d --> c, d, h, w
        image_series_tensor = torch.from_numpy(image_series).permute(2, 0, 1).unsqueeze(0)
        image_series_tensor *= 2.
        image_series_tensor -= 1.

        labels = torch.tensor((self.train_label[index]), dtype=torch.float32)

        return image_series_tensor, labels


class FFInfDataset(Dataset):
    def __init__(self, vid_path: Path) -> None:
        super().__init__()
        self.vid_path = vid_path
        self.cine_images = pims.open(self.vid_path.as_posix())

    def __len__(self):
        return len(self.cine_images) - 3

    def __getitem__(self, index: int) -> Tensor:
        # d, w, h --> w, h, d
        image_series = np.asarray(self.cine_images[index:index + 3], np.float32)
        image_series = normalize_cine(image_series)

        # w, h, d --> c, d, h, w
        image_series_tensor = torch.from_numpy(image_series).unsqueeze(0)
        image_series_tensor *= 2.
        image_series_tensor -= 1.

        return image_series_tensor


def normalize_cine(arr: np.ndarray) -> np.ndarray:
    """This normalizes an array to values between -1 and 1.

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    ndarray of float
        normalized array
    """
    ptp = arr.max(axis=(1,2)) - arr.min(axis=(1,2))
    # Handle edge case of a flat image.

    scaled_arr = (arr - arr.min(axis=(1,2)).reshape(-1, 1, 1)) / ptp.reshape(-1, 1, 1)

    return scaled_arr
