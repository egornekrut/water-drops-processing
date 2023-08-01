import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from albumentations import Compose, Crop, Flip, PadIfNeeded
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from src.segmentation.model import setup_segmentation_model
from src.utils.config import get_config_from_path
from src.utils.io import str_to_path


class CenterDetection:
    def __init__(self, config, prob_thres=None) -> None:
        self.config = config
        self.config.prob_thres = prob_thres if prob_thres else self.config.prob_thres

        self.config.device = 'cpu' if not torch.cuda.is_available() else self.config.device

        self.model = setup_segmentation_model(config, True)
        self.image_extensions = ['.png', '.jpeg', '.jpg']
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

    def run(
            self,
            input: Path,
            out_dir: Path,
            start_frame: int = 0,
            frame_step: int = 1,
            end_frame: int = -1,
            save_orig_frames: bool = True,
        ):
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        if input.is_dir():
            self.process_dir(input, out_dir)
        elif input.is_file() and input.suffix == '.mp4':
            self.process_video(
                input,
                out_dir,
                start_frame,
                frame_step,
                end_frame,
            )
        elif input.is_file() and input.suffix in self.image_extensions:
            self.process_image(input, out_dir)
        else:
            print(f'Cannot proceed {input.name}!')

    def process_video(
        self,
        vid_path: Path,
        out_dir: Path,
        start_frame: int = 0,
        frame_step: int = 1,
        end_frame: int = -1,
        save_orig_frames: bool = True,
    ):
        (out_dir / 'original_crops').mkdir(exist_ok=True, parents=True)
        (out_dir / 'mask_crops').mkdir(exist_ok=True, parents=True)
        (out_dir / 'masks').mkdir(exist_ok=True, parents=True)

        if save_orig_frames:
                (out_dir / 'original').mkdir(exist_ok=True, parents=True)

        if start_frame >= end_frame:
            raise ValueError

        video = cv2.VideoCapture(vid_path.as_posix())

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame > total_frames:
            end_frame = total_frames

        for fno in tqdm(range(start_frame, end_frame, frame_step)):
            video.set(cv2.CAP_PROP_POS_FRAMES, fno)
            _, frame = video.read()
            frame = frame[..., ::-1]

            if save_orig_frames:
                Image.fromarray(frame).save(out_dir / 'original' / (vid_path.stem + f'_frame_{fno}_crop.png'))

            mask_pil, cropped_mask, cropped_image = self.inference_image(frame)
            cropped_image.save(out_dir / 'original_crops' / (vid_path.stem + f'_frame_{fno}_crop.png'))
            cropped_mask.save(out_dir / 'mask_crops' / (vid_path.stem + f'_frame_{fno}_crop_mask.png'))
            mask_pil.save(out_dir / 'masks' / (vid_path.stem + f'_frame_{fno}_mask.png'))

    def process_dir(self, in_dir: Path, out_dir: Path):
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        for file in in_dir.glob('*.*'):
            if not file.suffix.lower() in self.image_extensions:
                continue
            self.process_image(file, out_dir)

    def process_image(self, file, out_dir):
        image = Image.open(file)
        mask_pil, cropped_mask, cropped_image = self.inference_image(image)

        cropped_image.save(out_dir / (file.stem + '_crop.' + file.suffix))
        cropped_mask.save(out_dir / (file.stem + '_crop_mask.' + file.suffix))
        mask_pil.save(out_dir / (file.stem + '_mask.' + file.suffix))

    def inference_image(self, image_pil: Union[np.ndarray, Image.Image]):
        if isinstance(image_pil, np.ndarray):
            image_pil = Image.fromarray(image_pil)
        image = np.asarray(image_pil.convert('L'))
        original_shape = image.shape

        transformed = self.transforms(image=image)
        img_tensor = transformed['image'].to(torch.float32) / 127.5 - 1

        probs = self.predict(img_tensor.unsqueeze(0))[..., :original_shape[0], :original_shape[1]].squeeze(0)
        mask = probs > self.config.prob_thres
        bbox = masks_to_boxes(mask)
        mask_pil = Image.fromarray(mask.squeeze(0).numpy().astype(np.uint8) * 255)
        cropped_mask = mask_pil.crop(bbox[0].numpy())
        cropped_image = image_pil.crop(bbox[0].numpy())



        return mask_pil, cropped_mask, cropped_image

    @torch.no_grad()
    def predict(self, image):
        return torch.sigmoid(self.model(image.to(self.config.device))).to('cpu')

    @staticmethod
    def decode(tensor: torch.Tensor) -> Image:
        tensor_denorm = torch.clip((tensor[0] + 1) * 127.5, min=0., max=255.).to(torch.uint8)
        pil_image = Image.fromarray(tensor_denorm.numpy())

        return pil_image


def parce_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True,
        type=str,
        help='A instance to process, should be video or folder.',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        type=str,
        help='A folder to save results.',
    )
    parser.add_argument(
        '--start-frame',
        default=0,
        type=int,
        help='Start frame in video to process.',
    )
    parser.add_argument(
        '--frame-step',
        default=1,
        type=int,
        help='Process every ith frame.',
    )
    parser.add_argument(
        '--end-frame',
        default=-1,
        type=int,
        help='End frame in video to process.',
    )
    parser.add_argument(
        '--save-orig-frames',
        default=True,
        type=bool,
        help='Is to save original video frames.',
    )
    parser.add_argument(
        '--config',
        default='./configs/default.py',
        type=str,
        help='Path to a config file.',
    )
    args = parser.parse_args()
    path_to_config = Path(args.config)

    config = get_config_from_path(path_to_config)
    config.inference_input = str_to_path(args.input, check_exist=True)
    config.inference_output = str_to_path(args.output_dir)
    config.start_frame = str_to_path(args.start_frame)
    config.frame_step = str_to_path(args.frame_step)
    config.end_frame = str_to_path(args.end_frame)
    config.save_orig_frames = str_to_path(args.save_orig_frames)

    return config


if __name__ == '__main__':
    config = parce_input_args()
    CenterDetection(config).run(
        config.inference_input,
        config.inference_output,
        config.start_frame,
        config.frame_step,
        config.end_frame,
        config.save_orig_frames,
    )
