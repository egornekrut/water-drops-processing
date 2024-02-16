import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import torch
from albumentations import Compose, PadIfNeeded
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.analysis.radius import ray_radius_estimator
from src.fframe.dataset import FFInfDataset
from src.fframe.model import FrameClassModel
from src.segmentation.model import setup_segmentation_model
from src.utils.config import get_config_from_path
from src.utils.io import str_to_path


class BubblesProcessor:
    def __init__(
        self,
        config=None,
        prob_thres=None,
        auto_frame: bool = True,
    ) -> None:
        if config:
            self.config = config
        else:
            self.config = get_config_from_path('./configs/default.py')
        # self.config.prob_thres = prob_thres if prob_thres else self.config.prob_thres

        self.config.device = 'cpu' if not torch.cuda.is_available() else self.config.device

        ### FFrame model
        self.auto_frame = auto_frame
        if self.auto_frame:
            fframe_model = FrameClassModel(1)
            fframe_model.load_state_dict(torch.load(self.config.fframe_ckpt_path, map_location='cpu'))
            fframe_model.to(self.config.device)
            fframe_model.eval()
            self.fframe_model = fframe_model

        self.ffprobs_plot = None

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
        input_arg: Path,
        out_dir: Path,
        start_frame: int = 0,
        frame_step: int = 1,
        end_frame: int = -1,
        save_orig_frames: bool = True,
        save_ext_pics: bool = True,
        save_plotted_results: bool = True,
        save_txt: bool = False,
        scale_px_mm: float = 1,
        auto_frame_first_offset: int = -10,
        auto_frame_last_offset: int = 200,
        auto_frame_thres: float = 0.5,
    ):
        self.statistics = {}

        if input_arg.is_dir():
            self.process_dir(
                input_arg,
                out_dir,
                start_frame,
                frame_step,
                end_frame,
                save_orig_frames,
                save_ext_pics,
                save_plotted_results,
                save_txt,
                scale_px_mm,
                auto_frame_first_offset,
                auto_frame_last_offset,
                auto_frame_thres,
            )

        elif input_arg.is_file() and input_arg.suffix == '.mp4':
            self.process_video(
                input_arg,
                out_dir,
                start_frame,
                frame_step,
                end_frame,
                save_orig_frames,
                save_ext_pics,
                save_plotted_results,
                save_txt,
                scale_px_mm,
            )
        elif input_arg.is_file() and input_arg.suffix == '.cine':
            self.process_cine_video(
                input_arg,
                out_dir,
                start_frame,
                frame_step,
                end_frame,
                save_orig_frames,
                save_ext_pics,
                save_plotted_results,
                save_txt,
                scale_px_mm,
                auto_frame_first_offset,
                auto_frame_last_offset,
                auto_frame_thres,
            )
        elif input_arg.is_file() and input_arg.suffix in self.image_extensions:
            raise NotImplementedError
            self.process_image(input_arg, out_dir)
        else:
            raise NotImplementedError
            print(f'Cannot proceed {input_arg.name}!')

    def setup_output_folders(
        self,
        out_dir: Path,
        save_orig_frames: bool = True,
        save_ext_pics: bool = True,
        save_plotted_results: bool = True,
        save_txt: bool = False,
    ) -> None:
        output_pics_folder = out_dir / 'pics'

        if save_orig_frames:
            self.original_frames_folder = output_pics_folder / 'original_frames'
            self.original_frames_folder.mkdir(exist_ok=True, parents=True)
        else:
            self.original_frames_folder = None

        if save_plotted_results:
            self.plotted_results_folder = output_pics_folder / 'plotted_results'
            self.plotted_results_folder.mkdir(exist_ok=True, parents=True)
        else:
            self.plotted_results_folder = None

        if save_ext_pics:
            self.orig_crops_folder = output_pics_folder / 'orig_crops'
            self.orig_crops_folder.mkdir(exist_ok=True, parents=True)
            self.mask_crops_folder = output_pics_folder / 'mask_crops'
            self.mask_crops_folder.mkdir(exist_ok=True, parents=True)
            self.output_boxed_masks_folder = output_pics_folder / 'boxed_masks'
            self.output_boxed_masks_folder.mkdir(exist_ok=True, parents=True)
            self.output_bubs_masks_folder = output_pics_folder / 'bubble_masks'
            self.output_bubs_masks_folder.mkdir(exist_ok=True, parents=True)
            self.mask_full_folder = output_pics_folder / 'full_masks'
            self.mask_full_folder.mkdir(exist_ok=True, parents=True)
        else:
            self.orig_crops_folder = None
            self.mask_crops_folder = None
            self.output_boxed_masks_folder = None
            self.output_bubs_masks_folder = None
            self.mask_full_folder = None

        if save_txt:
            self.txt_folder = output_pics_folder / 'txt_labels'
            self.txt_folder.mkdir(exist_ok=True, parents=True)
        else:
            self.txt_folder = None

    @staticmethod
    def xywh_xyxy(box, img_size):
        return ((box[0] - box[2] / 2) * img_size[0], (box[1] - box[3] / 2) * img_size[1], (box[0] + box[2] / 2) * img_size[0], (box[1] + box[3] / 2) * img_size[1])

    def saver(self, answer: Dict, no, scale_px_mm: float = 1., ruptures_stat: bool = True):
        self.statistics[no] = {}

        for key, elem in answer.items():
            if 'Диаметр' in key:
                self.statistics[no][key] = elem * scale_px_mm
            if 'Площадь' in key:
                self.statistics[no][key] = elem * (scale_px_mm ** 2)

        if ruptures_stat:
            self.statistics[no]['Число разрывов'] = len(answer['ruptures_stat'])
            self.statistics[no]['Площадь разрывов'] = sum([rupt['size_px'] for rupt in answer['ruptures_stat']]) * (scale_px_mm ** 2)

        if 'cropped_bubbles_masked' in answer:
            bubbles_mask = answer['cropped_bubbles_masked']
    
            bboxes = self.get_bbox_from_mask(bubbles_mask)
            self.statistics[no]['Число пузырей в РЗ'] = len(bboxes)
            self.statistics[no]['Площадь пузырей в РЗ'] = (np.asarray(bubbles_mask.convert('L')) > 0).sum() * (scale_px_mm ** 2)

            if self.output_bubs_masks_folder:
                bubbles_mask.save(self.output_bubs_masks_folder / ('' + f'frame_{no}_masked_bubles.png'))

            if self.output_boxed_masks_folder is not None:
                bubbles_mask_new = bubbles_mask.copy().convert('RGB')
                orig_crop = answer['orig_zone_crop'].convert('RGB')

                draw_orig = ImageDraw.Draw(orig_crop)
                draw_mask = ImageDraw.Draw(bubbles_mask_new)
        
                for box in bboxes:
                    draw_mask.rectangle(box, outline=(255, 0, 0, 125))
                    draw_orig.rectangle(box, outline=(255, 0, 0, 125))
        
                bubbles_mask_new.save(self.output_boxed_masks_folder / ('' + f'frame_{no}_masked_boxes.png'))
                orig_crop.save(self.output_boxed_masks_folder / ('' + f'frame_{no}_orig_bubbles.png'))
        else:
            self.statistics[no]['Число пузырей в РЗ'] = 0

        if 'plotted_results' in answer and self.plotted_results_folder:
            answer['plotted_results'].save(self.plotted_results_folder / f'frame_{no}_plot.png')

        if 'full_image' in answer and self.original_frames_folder:
            answer['full_image'].save(self.original_frames_folder / f'frame_{no}_original.png')

        if 'full_mask' in answer and self.mask_full_folder:
            answer['full_mask'].save(self.mask_full_folder / f'frame_{no}_mask.png')

        if 'cropped_image' in answer and self.orig_crops_folder:
            answer['cropped_image'].save(self.orig_crops_folder / f'frame_{no}_original_crop.png')

        if 'cropped_mask' in answer and self.mask_crops_folder:
            answer['cropped_mask'].save(self.mask_crops_folder /  f'frame_{no}_mask_crop.png')

    def process_video(
        self,
        vid_path: Path,
        out_dir: Path,
        start_frame: int = 0,
        frame_step: int = 1,
        end_frame: int = -1,
        save_orig_frames: bool = True,
        save_ext_pics: bool = True,
        save_plotted_results: bool = True,
        save_txt: bool = False,
        scale_px_mm: float = 1.,
    ):  
        self.setup_output_folders(
            out_dir,
            save_orig_frames,
            save_ext_pics,
            save_plotted_results,
            save_txt,
        )
        video = cv2.VideoCapture(vid_path.as_posix())

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame > total_frames:
            raise ValueError(f'Cannot find start frame No {start_frame} with max {total_frames}!')

        if end_frame == -1:
            end_frame = total_frames
        elif end_frame > total_frames:
            end_frame = total_frames

        print(f'Start processing video from {start_frame} to {end_frame} frame with step {frame_step}.')
        for fno in tqdm(range(start_frame, end_frame, frame_step)):
            video.set(cv2.CAP_PROP_POS_FRAMES, fno)
            _, frame = video.read()
            frame = frame[..., ::-1]

            if save_txt:
                txt_save_path = self.txt_folder / f'{vid_path.stem}_frame_{fno}.txt'
            else:
                txt_save_path = None
            answer = self.process_img_step1(frame, txt_save_path=txt_save_path, save_plotted_results=save_plotted_results)
            self.saver(answer, fno, scale_px_mm)
    
        pd.DataFrame.from_dict(self.statistics, orient='index').to_excel(out_dir / f'{vid_path.stem}_stat.xlsx')

    def process_dir(
        self,
        in_dir: Path,
        out_dir: Path,
        start_frame: int = 0,
        frame_step: int = 1,
        end_frame: int = -1,
        save_orig_frames: bool = True,
        save_ext_pics: bool = True,
        save_plotted_results: bool = True,
        save_txt: bool = False,
        scale_px_mm: float = 1.,
        auto_frame_first_offset: int = -10,
        auto_frame_last_offset: int = 200,
        auto_frame_thres: float = 0.5,
    ):
        for file in in_dir.glob('*.*'):
            if file.suffix.lower() == '.cine':
                vid_out_dir = out_dir / file.stem
                vid_out_dir.mkdir(parents=True, exist_ok=False)
                self.process_cine_video(
                    file,
                    vid_out_dir,
                    start_frame,
                    frame_step,
                    end_frame,
                    save_orig_frames,
                    save_ext_pics,
                    save_plotted_results,
                    save_txt,
                    scale_px_mm,
                    auto_frame_first_offset,
                    auto_frame_last_offset,
                    auto_frame_thres,
                )
    # def process_image(self, file, out_dir, save_txt=False):
    #     image = Image.open(file)

    #     if save_txt:
    #         txt_save_path = out_dir / (file.stem + '.txt')
    #     else:
    #         txt_save_path = None

    #     mask_pil, cropped_mask, cropped_image = self.inference_image(image, txt_save_path=txt_save_path)

    #     cropped_image.save(out_dir / (file.stem + '_crop.' + file.suffix))
    #     cropped_mask.save(out_dir / (file.stem + '_crop_mask.' + file.suffix))
    #     mask_pil.save(out_dir / (file.stem + '_mask.' + file.suffix))
    
    @torch.inference_mode()
    def find_first_frame(self, vid_path: Path, ff_thres = 0.9):
        ff_dataset = FFInfDataset(vid_path=vid_path)
        all_probs = [0, 0]
        fframe_res = 0
        itetator = tqdm(range(2, len(ff_dataset)), total=len(ff_dataset))

        for idx in itetator:
            frame_pack = ff_dataset[idx]
            itetator.set_description(f'Выполняется поиск контакта')
            
            with torch.autocast(device_type=self.config.device):
                prob = self.fframe_model(frame_pack.unsqueeze(0).to(self.config.device)).sigmoid().to(device='cpu', dtype=torch.float32).squeeze().item()
            all_probs.append(prob)

            if prob > ff_thres:
                fframe_res = idx
                break

        like_hood_ff = np.argmax(all_probs)
        self.ffprobs_plot = plt.plot(all_probs, label='Вероятности')
        self.ffprobs_plot = plt.vlines(x = like_hood_ff, ymin=0, ymax=max(all_probs) + 0.1, colors='red', label=f'Предполагаемый момент контакта: {np.argmax(all_probs)}, p = {np.max(all_probs):.2f}', ls=':', lw=2)
        plt.xlabel('Кадры')
        plt.ylabel('Вероятность')
        plt.title('Распределение вероятностей момента контакта')
        plt.legend()

        return fframe_res

    def process_cine_video(
        self,
        vid_path: Path,
        out_dir: Path,
        start_frame: int = 0,
        frame_step: int = 1,
        end_frame: int = -1,
        save_orig_frames: bool = True,
        save_ext_pics: bool = True,
        save_plotted_results: bool = True,
        save_txt: bool = False,
        scale_px_mm: float = 1.,
        auto_frame_first_offset: int = -10,
        auto_frame_last_offset: int = 200,
        auto_frame_thres: float = 0.5,
    ):  
        self.setup_output_folders(
            out_dir,
            save_orig_frames,
            save_ext_pics,
            save_plotted_results,
            save_txt,
        )

        cine_images = pims.open(vid_path.as_posix())

        if self.auto_frame and start_frame == 0:
            start_frame = self.find_first_frame(vid_path, auto_frame_thres)
            if start_frame != 0:
                print(f'Момент контакта был автоматически обнаружен: {start_frame}.')
                start_frame = max(start_frame + auto_frame_first_offset, 0)
                end_frame = min(start_frame + auto_frame_last_offset, len(cine_images))
                print(f'Итоговый интервал кадров для обработки: {start_frame} - {end_frame}.')
            else:
                print('Первый кадр не был обнаружен. Перезапустите процесс с ручной настройкой или понизьте порог для момента контакта!')
                return None

        if end_frame == -1:
            end_frame = len(cine_images)

        iterator = tqdm(range(start_frame, end_frame, frame_step), total=(end_frame - start_frame) // frame_step)

        for enum, idx in enumerate(iterator):
            # grayscale img with (h, w) dimensions
            uint8_img = cv2.normalize(np.asarray(cine_images[idx]), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            if save_txt:
                txt_save_path = self.txt_folder / f'{vid_path.stem}_frame_{idx}.txt'
            else:
                txt_save_path = None

            answer = self.process_img_step1(
                uint8_img,
                txt_save_path=txt_save_path,
                save_plotted_results=save_plotted_results,
                ruptures_stat=True,
            )

            self.saver(answer, idx, scale_px_mm)
    
        pd.DataFrame.from_dict(self.statistics, orient='index').to_excel(out_dir / f'{vid_path.stem}_stat.xlsx')

    def process_img_step1(
            self,
            image_pil: Union[np.ndarray, Image.Image],
            **kwargs,
        ) -> Dict:
        if isinstance(image_pil, np.ndarray):
            image_pil = Image.fromarray(image_pil)

        yolo_answer = self.step_1_model(
            image_pil,
            verbose=False,
            conf=self.config.step1_thres,
            retina_masks=True,
            device=self.config.device,
        )
        return self.gather_yolo_results(yolo_answer[0], image_pil, **kwargs)

    @staticmethod
    def get_bbox_from_mask(img: Image.Image) -> List[Tuple[float, ...]]:
        img_np = np.asarray(img)
        contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_wh = img.size
        all_bboxes = []
        shapes = []
        for i in range(len(contours)):
            shapes.append(contours[i].shape[1])
            mined = contours[i][:, 0, :].min(0)
            maxed = contours[i][:, 0, :].max(0)
            bbox_coords = (mined[0], mined[1], maxed[0], maxed[1])

            crop = img_np[bbox_coords[1]:bbox_coords[3] + 1, bbox_coords[0]:bbox_coords[2] + 1]
            if (crop != 0).sum() < 5:
                continue

            # shape_norm = (
            #     (bbox_coords[2] + bbox_coords[0]) / (2 * img_wh[0]),
            #     (bbox_coords[3] + bbox_coords[1]) / (2 * img_wh[1]),
            #     (bbox_coords[2] - bbox_coords[0]) / img_wh[0],
            #     (bbox_coords[3] - bbox_coords[1]) / img_wh[1],
            # )
            all_bboxes.append(bbox_coords)

        return all_bboxes

    def gather_yolo_results(self, yolo_answer: Results, image_pil: Image.Image, **kwargs):
        answer = {}
        pic_size = image_pil.size
        hwc_size = (*pic_size[::-1], 3)
        classes = yolo_answer.boxes.cls

        if kwargs.get('save_plotted_results', False):
            plotted_results = yolo_answer.plot()[..., ::-1]
            answer['plotted_results'] = Image.fromarray(plotted_results)

        # Трекаем число и размеры разрывов, добавляем в лист
        answer['ruptures_stat'] = []

        full_mask = np.zeros(hwc_size[:2], dtype=np.uint8)
        ruptures = np.zeros(hwc_size, dtype=np.uint8)
        color_pic = np.zeros(hwc_size, dtype=np.uint8)
        bboxes = {}
        # TODO: ДОБАВИТЬ СОВМЕЩЕНИЕ МАСОК, ЕСЛИ НАШЛОСЬ НЕСКОЛЬКО ОБЪЕКТОВ

        if len(classes):
            if kwargs.get('txt_save_path', False):
                yolo_answer.save_txt(kwargs.get('txt_save_path'))
    
            masks = yolo_answer.masks.data
    
            for cls_enum, cls in enumerate(classes):
                cls_int = int(cls)
                single_mask = masks[cls_enum]
                pic_channel = (single_mask == 1).cpu().numpy().astype(dtype=np.uint8)

                color_pic[..., cls_int] += pic_channel * 255

                bboxes[cls_int] = yolo_answer.boxes.xyxy[cls_enum].cpu().numpy()

                if cls_int == 0 and pic_channel.sum():
                    n_radius = kwargs.get('n_radius', 32)
                    answer[f'Диаметр_лучи_{n_radius}'] = ray_radius_estimator(pic_channel * 255, n_radius)
                    answer['Диаметр_бокс'] = float(yolo_answer.boxes.xywh[cls_int, 2:].mean().cpu())
                    answer['Диаметр_pir2'] = 2 * np.sqrt(pic_channel.sum() / np.pi)

                elif cls_int == 1 and pic_channel.sum():
                    bounding_box = [int(np.round(i)) for i in bboxes[cls_int]]
                    orig_zone_crop = image_pil.crop(bounding_box).convert('L')

                    answer['orig_zone_crop'] = orig_zone_crop
                    bubbles = self.draw_bubbles(orig_zone_crop)

                    # full_mask[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]] = bubbles
                    # answer['full_bubbles'] = Image.fromarray(full_mask)
                    answer['cropped_bubbles'] = Image.fromarray(bubbles)
                    mask_zone = pic_channel[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
                    answer['zone_mask'] = Image.fromarray(mask_zone * 255)

                    masked_bubs = bubbles * mask_zone
                    answer['cropped_bubbles_masked'] = Image.fromarray(masked_bubs)

                elif cls_int == 2 and pic_channel.sum():
                    # Разрывы
                    ruptures[..., 0] += pic_channel * 255
                    ruptures[..., 2] += pic_channel * 255
                    answer['ruptures_stat'].append(
                        {
                            'size_px': pic_channel.sum(),
                            'height': yolo_answer.boxes.xywh[cls_enum].cpu().numpy()[3],
                            'width': yolo_answer.boxes.xywh[cls_enum].cpu().numpy()[2],
                        }
                    )

                    # bounding_box = [int(np.round(i)) for i in bboxes[int(cls)]]
                    # orig_zone_crop = image_pil.crop(bounding_box).convert('L')

                    # bubbles = self.draw_bubbles(orig_zone_crop)

                    # # full_mask[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]] = bubbles
                    # # answer['full_bubbles'] = Image.fromarray(full_mask)
                    # answer['cropped_bubbles'] = Image.fromarray(bubbles)
                    # mask_zone = pic_channel[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
                    # answer['zone_mask'] = Image.fromarray(mask_zone * 255)

                    # masked_bubs = bubbles * mask_zone
                    # answer['cropped_bubbles_masked'] = Image.fromarray(masked_bubs)

        color_pic[ruptures.sum(axis=-1) > 0] = (255, 0, 255)
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
        
        answer['cropped_mask'] = cropped_mask
        answer['cropped_image'] = cropped_image
        answer['full_image'] = full_image
        answer['full_mask'] = mask_pil

        return answer

    @torch.no_grad()
    def draw_bubbles(self, image: Image.Image):
        # TODO: Add center dynamic padding
        image_np = np.asarray(image)
        image_shape = image_np.shape
        transformed = self.transforms(image=image_np)
        img_tensor = transformed['image'].to(torch.float32) / 127.5 - 1
        model_output = self.step_2_model(img_tensor.to(self.config.device).unsqueeze(0)).cpu().squeeze((0, 1)).numpy()
        model_output_orig_size = model_output[:image_shape[0], :image_shape[1]]
        bubble_mask = (model_output_orig_size > self.config.step2_thres).astype(np.uint8) * 255

        return bubble_mask

    @staticmethod
    def decode(tensor: torch.Tensor) -> Image.Image:
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
        '--save-ext-pics',
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
    
    config.save_ext_pics = args.save_ext_pics
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
        config.save_ext_pics,
        config.save_plots,
        config.save_txt,
    )
