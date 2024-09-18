from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pims
import torch
from easydict import EasyDict
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.analysis.radius import ray_radius_estimator
from src.fframe.model import FrameClassModel
from src.utils.config import get_config_from_path
from src.utils.inference_v2 import BasicModelPipeline, BasicProcessor


class YoloDetectorModel(BasicModelPipeline):
    def __init__(
            self,
            model_config,
            device: Optional[str] = None,
    ) -> None:
        super().__init__(model_config, ['image'], ['full_mask'], device)
        self.num_radius = 32

    def _setup_model(self) -> YOLO:
        return YOLO(model=self.model_config.weights, task='segment')

    def _model_inference(
        self,
        image_pil: Image.Image,
    ) -> Dict[str, Any]:
        yolo_res: Results = self.model.track(
            image_pil,
            persist=True,
            **self.model_config['inference_config'],
        )[0]

        return self._gather_yolo_results(yolo_res, image_pil)
    
    def _gather_yolo_results(self, yolo_answer: Results, image_pil: Image.Image) -> Dict[str, Any]:
        """Post-processing of the results from YOLO model.

        Args:
            yolo_answer (Results): Results from YOLO model
            image_pil (Image.Image): Original image in PIL

        Returns:
            Dict[str, Any]: Post-processed results in dictionary form
        """
        answer = {}
        pic_size = image_pil.size
        hwc_size = (*pic_size[::-1], 3)
        classes = yolo_answer.boxes.cls

        answer['plotted_results'] = Image.fromarray(yolo_answer.plot())

        # Трекаем число и размеры разрывов, добавляем в лист
        answer['ruptures_stat'] = []

        full_mask = np.zeros(hwc_size[:2], dtype=np.uint8)
        ruptures = np.zeros(hwc_size, dtype=np.uint8)
        color_pic = np.zeros(hwc_size, dtype=np.uint8)
        bboxes = {}

        if len(classes):
            masks = yolo_answer.masks.data
    
            for cls_enum, cls in enumerate(classes):
                cls_int = int(cls)
                single_mask = masks[cls_enum]
                pic_channel = (single_mask == 1).cpu().numpy().astype(dtype=np.uint8)

                color_pic[..., cls_int] += pic_channel * 255
                bboxes[cls_int] = yolo_answer.boxes.xyxy[cls_enum].cpu().numpy()

                if cls_int == 0 and pic_channel.sum():
                    # Капля
                    answer[f'diam_rays_{self.num_radius}'] = ray_radius_estimator(
                        pic_channel * 255,
                        self.num_radius,
                    )
                    answer['diam_pir2'] = 2 * np.sqrt(pic_channel.sum() / np.pi)

                elif cls_int == 1 and pic_channel.sum():
                    # Внутренность капли
                    bounding_box = [int(np.round(i)) for i in bboxes[cls_int]]
                    mask_zone = pic_channel[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
                    answer['zone_mask'] = Image.fromarray(mask_zone * 255)

                elif cls_int == 2 and pic_channel.sum():
                    # Разрывы
                    ruptures[..., 2] += pic_channel * 255
                    answer['ruptures_stat'].append(
                        {
                            'size_px': pic_channel.sum(),
                            'height': yolo_answer.boxes.xywh[cls_enum].cpu().numpy()[3],
                            'width': yolo_answer.boxes.xywh[cls_enum].cpu().numpy()[2],
                        },
                    )

        color_pic[ruptures.sum(axis=-1) > 0] = (0, 0, 255)
        full_mask_pil = Image.fromarray(color_pic)

        if 0 in bboxes is not None:
            cropped_mask = full_mask_pil.crop(bboxes[0])
            cropped_image = image_pil.crop(bboxes[0])
        else:
            cropped_mask = full_mask_pil
            cropped_image = image_pil

        if full_mask.sum():
            full_image = Image.blend(image_pil, Image.fromarray(full_mask), 0.5)
        else:
            full_image = image_pil
        
        answer['cropped_mask'] = cropped_mask
        answer['cropped_image'] = cropped_image
        answer['full_image'] = full_image
        answer['full_mask'] = full_mask_pil

        return answer


class ContactFinderModel(BasicModelPipeline):
    def __init__(
        self,
        model_config,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_config, ['video_path'], ['is_contact'], device)
        self.cf_thres = model_config['thres']

    def _setup_model(self) -> FrameClassModel:
        model = FrameClassModel(1, self.model_config['weights'])
        model.eval()
        return model

    def _preprocess(self, input_data: Dict[str, Iterable]) -> Iterable:
        """Preprocess input data before passing to the model.

        Args:
            input_data (Any): Data to be processed by the model

        Returns:
            Any: Processed data
        """
        return input_data['image_sequence']

    def _model_inference(
        self,
        image_sequence: Iterable,
    ) -> Dict[str, Any]:
        cf_frame = 0
        cf_probs = [0., 0.]

        iterator = tqdm(range(2, len(image_sequence) - 2), total=len(image_sequence) - 2)

        for idx in iterator:
            image_series = np.asarray(image_sequence[idx - 2:idx + 3], np.float32)
            image_series = self.normalize_cine(image_series) * 2 - 1
            image_series_tensor = torch.from_numpy(image_series).unsqueeze(0).unsqueeze(0).to(device=self.device)

            cf_res: torch.Tensor = self.model.forward(
               image_series_tensor,
            )
            cf_prob = cf_res.sigmoid().item()
            cf_probs.append(cf_prob)

            if cf_prob > self.cf_thres:
                cf_frame = idx
                break
    
        return {'is_contact': cf_frame, 'cf_probs': cf_probs}

    @staticmethod
    def normalize_cine(arr: np.ndarray) -> np.ndarray:
        """This normalizes an array to values between 0 and 1."""
        ptp = arr.max(axis=(1,2)) - arr.min(axis=(1,2))
        # Handle edge case of a flat image.

        scaled_arr = (arr - arr.min(axis=(1,2)).reshape(-1, 1, 1)) / ptp.reshape(-1, 1, 1)

        return scaled_arr


class VideoProcessor(BasicProcessor):
    def __init__(
        self,
        config,
        result_dir: Optional[Union[str, Path]] = None,
        make_video: bool = False,
        save_all_pics: bool = False,
        maximum_frame_count: int = 500,
        contact_frame_offset: int = -5,
    ) -> None:
        super().__init__(config, result_dir)
        self.curr_video_format = None

        if not self.result_dir:
            make_video = False
            save_all_pics = False

        self.make_video = make_video
        self.save_all_pics = save_all_pics

        self.maximum_frame_count = maximum_frame_count
        self.contact_frame_offset = contact_frame_offset

        if self.config.get('contact_frame_model'):
            self.cf_model = ContactFinderModel(self.config['contact_frame_model'], device=self.device)
        else:
            self.cf_model = None

    def set_formats(self):
        return ('.mp4', '.cine')

    def _set_saving_rule(self) -> Dict[str, Path]:
        rules = {}

        if self.result_dir is None:
            print(f'result_dir is not set! This prevents video creation and other saving.\nTo prevent this later pass result_dir to the init.')
            return rules

        exp_path = Path(self.result_dir) / self.exp_name

        if self.save_all_pics:
            rules.update({
                'image': {'path': exp_path / 'original_frames'},
                'full_mask': {'path': exp_path / 'full_masks'},
            })
        
        if self.make_video:
            rules['result_video'] = {'save_func': self._blend_image_for_video}
            self.video_writer = cv2.VideoWriter(
                (exp_path / f'{self.exp_name}_result.avi').as_posix(),
                cv2.VideoWriter_fourcc(*'MPEG'),
                10,
                (640, 480),
            )

        return rules

    def setup_model_pipeline(self) -> List[BasicModelPipeline]:
        return [
            YoloDetectorModel(self.config.yolo_configuration, device=self.config.device),
        ]

    def _open_file(self, input_file: Path) -> Dict[str, Any]:
        metadata_dict: Dict[str, Any] = {'input_file': input_file.as_posix()}

        if input_file.suffix == '.mp4':
            self.curr_video_format = 'mp4'
            # frame_stream = cv2.VideoCapture(vid_path.as_posix())
            raise NotImplementedError

        elif input_file.suffix == '.cine':
            self.curr_video_format = 'cine'
            frame_stream = pims.open(input_file.as_posix())
        else:
            raise ValueError(f'Unsupported file type {input_file.stem}')
        start_frame, end_frame = self._find_video_bounds(frame_stream)

        metadata_dict['stream'] = frame_stream[start_frame:end_frame]
        metadata_dict['stream_len'] = len(metadata_dict['stream'])

        return metadata_dict
    
    def _find_video_bounds(self, frame_stream: Iterable) -> Tuple[int, int]:
        start_frame = 0
        end_frame = self.maximum_frame_count

        if self.cf_model is not None:
            print('Looking for contact frame...')
            start_frame: int = self.cf_model({'image_sequence': frame_stream})['is_contact']

        if start_frame:
            print(f'Successfuly found contact frame {start_frame}! Video starts from {start_frame + self.contact_frame_offset}')
            start_frame += self.contact_frame_offset
        else:
            print('Contact frame not found! Processing starts from 0 frame.')

        if len(frame_stream) - start_frame > self.maximum_frame_count:
            print(f'Clip video frames from {len(frame_stream) - start_frame} to {self.maximum_frame_count}.\nTo prevent this adjust maximum_frame_count variable.')
            end_frame += start_frame

        return start_frame, end_frame

    def _preprocess_input(self, frame: np.ndarray) -> Dict[str, Any]:
        """Preprocess the input to be processed by the model pipeline.

        Args:
            frame (np.ndarray): Frame to be processed by the model pipeline.

        Returns:
            Dict[str, Any]: Preprocessed frame to be processed by the model pipeline.
        """
        if self.curr_video_format == 'cine':
            normed_img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            normed_img = frame

        return {'image': Image.fromarray(normed_img)}

    def _get_final_result(self, states: Dict[str, Any]) -> Dict[str, Any]:
        if self.video_writer is not None:
            cv2.destroyAllWindows()
            self.video_writer.release()
            self.video_writer = None
            print('Video with segmentation result saved.')

        return states

    def _blend_image_for_video(self, state: Dict[str, Image.Image], frame_idx: int) -> bool:
        if self.video_writer is None:
            return False

        blended_frame = Image.blend(state['image'].convert('RGB'), state['full_mask'].convert('RGB'), 0.2)
        ImageDraw.Draw(blended_frame).text(
            (0, 0),
            f'Frame {frame_idx}',
            (255, 255, 255),
        )

        # Save the image as a frame of a video
        self.video_writer.write(
            np.asarray(blended_frame)[..., ::-1],
        )
        return True
