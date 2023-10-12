import argparse
from pathlib import Path
from typing import Optional, Union
from ultralytics import YOLO
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


class BubblesProcessor:
    def __init__(self, config, prob_thres=None) -> None:
        self.config = config
        self.config.prob_thres = prob_thres if prob_thres else self.config.prob_thres

        self.config.device = 'cpu' if not torch.cuda.is_available() else self.config.device

        # Instance Segmentation model
        self.step_1_model = YOLO(self.config.step_1_ckpt_path)
        self.step_2_model = setup_segmentation_model(self.config, True)

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
            save_crops: bool = True,
            save_plotted_results: bool = True,
            save_txt: bool = False,
        ):
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        if input.is_dir():
            raise NotImplementedError
            self.process_dir(input, out_dir)

        elif input.is_file() and input.suffix == '.mp4':
            self.process_video(
                input,
                out_dir,
                start_frame,
                frame_step,
                end_frame,
                save_orig_frames,
                save_crops,
                save_plotted_results,
                save_txt,
            )
        elif input.is_file() and input.suffix in self.image_extensions:
            raise NotImplementedError
            self.process_image(input, out_dir)
        else:
            raise NotImplementedError
            print(f'Cannot proceed {input.name}!')

    def process_video(
        self,
        vid_path: Path,
        out_dir: Path,
        start_frame: int = 0,
        frame_step: int = 1,
        end_frame: int = 0,
        save_orig_frames: bool = True,
        save_crops: bool = True,
        save_plotted_results: bool = True,
        save_txt: bool = False,
    ):  
        output_masks_folder = out_dir / 'masks'
        output_masks_folder.mkdir(exist_ok=True, parents=True)

        if save_orig_frames:
            original_frames_folder = out_dir / 'original_frames'
            original_frames_folder.mkdir(exist_ok=True, parents=True)

        if save_plotted_results:
            plotted_results_folder = out_dir / 'plotted_results'
            plotted_results_folder.mkdir(exist_ok=True, parents=True)

        if save_crops:
            orig_crops_folder = out_dir / 'orig_crops'
            orig_crops_folder.mkdir(exist_ok=True, parents=True)
            mask_crops_folder = out_dir / 'mask_crops'
            mask_crops_folder.mkdir(exist_ok=True, parents=True)
        
        if save_txt:
            txt_folder = out_dir / 'txt_labels'

        video = cv2.VideoCapture(vid_path.as_posix())

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame == 0:
            end_frame = total_frames

        if start_frame > total_frames:
            raise ValueError(f'Cannot find start frame No {start_frame} with max {total_frames}!')

        if end_frame > total_frames:
            end_frame = total_frames

        print(f'Start processing video from {start_frame} to {end_frame} frame with step {frame_step}.')

        for fno in tqdm(range(start_frame, end_frame, frame_step)):
            video.set(cv2.CAP_PROP_POS_FRAMES, fno)
            _, frame = video.read()
            frame = frame[..., ::-1]

            if save_txt:
                txt_save_path = txt_folder / f'{vid_path.stem}_frame_{fno}.txt'
            else:
                txt_save_path = None

            mask_pil, cropped_mask, cropped_image, plotted_results, bub_image = self.inference_image(frame, txt_save_path=txt_save_path)

            if save_plotted_results:
                plotted_results.save(plotted_results_folder / (vid_path.stem + f'_frame_{fno}_plot.png'))
            if save_crops:
                cropped_mask.save(mask_crops_folder / (vid_path.stem + f'_frame_{fno}_mask_crop.png'))
                cropped_image.save(orig_crops_folder / (vid_path.stem + f'_frame_{fno}_orig_crop.png'))

            if save_orig_frames:
                Image.fromarray(frame).save(original_frames_folder / (vid_path.stem + f'_frame_{fno}.png'))
                
            mask_pil.save(output_masks_folder / (vid_path.stem + f'_frame_{fno}_mask.png'))
            bub_image.save(output_masks_folder / (vid_path.stem + f'_frame_{fno}_bubles.png'))

    def process_dir(self, in_dir: Path, out_dir: Path):
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        for file in in_dir.glob('*.*'):
            if not file.suffix.lower() in self.image_extensions:
                continue
            self.process_image(file, out_dir)

    def process_image(self, file, out_dir, save_txt=False):
        image = Image.open(file)

        if save_txt:
            txt_save_path = out_dir / (file.stem + '.txt')
        else:
            txt_save_path = None

        mask_pil, cropped_mask, cropped_image = self.inference_image(image, txt_save_path=txt_save_path)

        cropped_image.save(out_dir / (file.stem + '_crop.' + file.suffix))
        cropped_mask.save(out_dir / (file.stem + '_crop_mask.' + file.suffix))
        mask_pil.save(out_dir / (file.stem + '_mask.' + file.suffix))

    def inference_image(
            self,
            image_pil: Union[np.ndarray, Image.Image],
            pic_size=(480, 640, 3),
            txt_save_path: Optional[Path] = None,
        ):
        if isinstance(image_pil, np.ndarray):
            image_pil = Image.fromarray(image_pil)

        # image = np.asarray(image_pil.convert('L'))
        yolo_results = self.step_1_model(image_pil, verbose=False, conf=self.config.prob_thres, retina_masks=True)[0]
        
        plotted_results = yolo_results.plot()[..., ::-1]
        classes = yolo_results.boxes.cls
        
        full_mask = np.zeros((*image_pil.size[::-1], 3), dtype=np.uint8)
        color_pic = np.zeros(pic_size, dtype=np.uint8)
        bboxes = {}

        if len(classes):
            if txt_save_path:
                yolo_results.save_txt(txt_save_path)

            masks = yolo_results.masks.data
            for cls_id, cls in enumerate(classes):
                single_mask = masks[cls_id]
                pic_channel = (single_mask == 1).cpu().numpy().astype(dtype=np.uint8) * 255
                color_pic[..., int(cls)] = pic_channel
                # if int(cls) == 0:
                bboxes[int(cls)] = yolo_results.boxes.xyxy[cls_id].cpu().numpy()

                if int(cls) == 1:
                    bounding_box = [int(i) for i in bboxes[int(cls)]]
                    probs = self.draw_bubbles(image_pil.crop(bounding_box).convert('L'))
                    bubble_mask = (probs > self.config.prob_thres).astype(np.uint8)[:bounding_box[2]-bounding_box[0], :bounding_box[3]-bounding_box[1]]

                    full_mask[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3], 0] = bubble_mask

        mask_pil = Image.fromarray(color_pic)

        if 0 in bboxes is not None:
            cropped_mask = mask_pil.crop(bboxes[0])
            cropped_image = image_pil.crop(bboxes[0])
        else:
            cropped_mask = mask_pil
            cropped_image = image_pil

        if full_mask.sum():
            full_image = Image.blend(image_pil, Image.fromarray(full_mask), 0.5)
        else:
            full_image = image_pil

        return mask_pil, cropped_mask, cropped_image, Image.fromarray(plotted_results), full_image

    @torch.no_grad()
    def draw_bubbles(self, image):
        transformed = self.transforms(image=np.asarray(image))
        img_tensor = transformed['image'].to(torch.float32) / 127.5 - 1
        model_output = self.step_2_model(img_tensor.to(self.config.device).unsqueeze(0)).to('cpu')

        return model_output.squeeze((0, 1)).numpy()

    @staticmethod
    def decode(tensor: torch.Tensor) -> Image:
        tensor_denorm = torch.clip((tensor[0] + 1) * 127.5, min=0., max=255.).to(torch.uint8)
        pil_image = Image.fromarray(tensor_denorm.numpy())

        return pil_image


def parce_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=False,
        default='/mnt/c/Users/egorn/Desktop/WDP/videos/4.6.mp4',
        type=str,
        help='A instance to process, should be video or folder.',
    )
    parser.add_argument(
        '--output-dir',
        required=False,
        default='./tmp_output',
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
        default=0,
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
        '--save-crops',
        default=False,
        type=bool,
        help='Is to save cropped version of frames.',
    )
    parser.add_argument(
        '--save-plots',
        default=False,
        type=bool,
        help='Is to save plots.',
    )
    parser.add_argument(
        '--save-txt',
        default=False,
        type=bool,
        help='Is to save predictions in txt Yolo format.',
    )
    parser.add_argument(
        '--config',
        default='./configs/default.py',
        type=str,
        help='Path to a config file.',
    )
    parser.add_argument(
        '--prob-thres',
        default=-1,
        type=float,
        help='Probability to draw mask.',
    )
    args = parser.parse_args()
    path_to_config = Path(args.config)

    config = get_config_from_path(path_to_config)
    config.inference_input = str_to_path(args.input, check_exist=True)
    config.inference_output = str_to_path(args.output_dir)

    config.start_frame = args.start_frame
    config.frame_step = args.frame_step
    config.end_frame = args.end_frame

    config.save_orig_frames = args.save_orig_frames
    config.prob_thres = args.prob_thres if args.prob_thres != -1 else config.prob_thres
    
    config.save_crops = args.save_crops
    config.save_plots = args.save_plots
    config.save_txt = args.save_txt

    
    return config


if __name__ == '__main__':
    config = parce_input_args()
    BubblesProcessor(config).run(
        config.inference_input,
        config.inference_output,
        config.start_frame,
        config.frame_step,
        config.end_frame,
        config.save_orig_frames,
        config.save_crops,
        config.save_plots,
        config.save_txt,
    )
