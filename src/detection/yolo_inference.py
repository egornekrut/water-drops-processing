from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pims
from torchvision.ops import box_iou
import torch
from easydict import EasyDict
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.analysis.radius import ray_diameter_estimator
from src.fframe.model import FrameClassModel
from src.utils.inference_v2 import BasicModelPipeline, BasicProcessor


class YoloDetectorModel(BasicModelPipeline):
    def __init__(
            self,
            model_config: EasyDict,
            device: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_config,
            ['image'],
            ['full_mask_bool', 'full_mask', 'ruptures_stat', 'droplet_stat', 'plot'],
            device,
        )

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
        classes = yolo_answer.boxes.cls.cpu().numpy().astype(np.int32)

        answer['ruptures_stat'] = {}
        answer['droplet_stat'] = {}

        all_masks = np.zeros(hwc_size, dtype=bool)

        if len(classes):
            masks = yolo_answer.masks.data.cpu().numpy()
            bboxes = yolo_answer.boxes.xywh.cpu().numpy()
            bbox_x1x2_torch = yolo_answer.boxes.xyxy
            for class_id in range(3):
                objects_indices = np.where(classes == class_id)[0]

                if len(objects_indices) == 0:
                    continue

                single_class_mask = masks[objects_indices].sum(axis=0) > 0
                all_masks[..., class_id] = single_class_mask

                if class_id == 0:
                    # Find the largest droplet
                    boxes_size = bboxes[objects_indices][:, 2:]
                    boxes_area = boxes_size[:, 0] * boxes_size[:, 1]
                    largest_id = np.argmax(boxes_area)
                    largest_mask = masks[objects_indices[largest_id]]

                    ray_diam, center_x, center_y = ray_diameter_estimator(largest_mask, 32)
                    answer['droplet_stat']['droplet_diam_rays_32'] = ray_diam
                    answer['droplet_stat']['droplet_diam_pir2'] = 2 * np.sqrt(single_class_mask.sum() / np.pi)
                    answer['droplet_stat']['droplet_mass_center'] = (center_x, center_y)

                elif class_id == 2:
                    # Ruptures
                    all_masks[..., 0] = all_masks[..., 0] & ~single_class_mask
                    unique_ids = yolo_answer.boxes.id[objects_indices].cpu().numpy()
                    rupture_masks = masks[objects_indices]
                    rupture_bboxes = bboxes[objects_indices]
                    
                    for curr_id, (unique_id, rupture_mask, rupture_bbox) in enumerate(zip(unique_ids, rupture_masks, rupture_bboxes)):
                        distance = np.linalg.norm(rupture_bbox[:2].reshape(1, -1) - rupture_bboxes[:, :2], axis=1)
                        distance[curr_id] = np.inf

                        answer['ruptures_stat'][int(unique_id)] = {
                            'px_area': (rupture_mask > 0).sum(),
                            'x_center': rupture_bbox[0],
                            'y_center': rupture_bbox[1],
                            'width': rupture_bbox[2],
                            'height': rupture_bbox[3],
                        }
                        if len(distance) > 1:
                            min_index = np.argmin(distance)
                            nearest_rupture_idx = int(unique_ids[min_index])
                            answer['ruptures_stat'][int(unique_id)]['nearest_rupture_id'] = nearest_rupture_idx
                            answer['ruptures_stat'][int(unique_id)]['nearest_rupture_iou'] = box_iou(
                                bbox_x1x2_torch[objects_indices][curr_id].unsqueeze(0),
                                bbox_x1x2_torch[objects_indices][min_index].unsqueeze(0),
                            ).item()

        answer['full_mask_bool'] = all_masks
        answer['full_mask'] = Image.fromarray(all_masks.astype(np.uint8) * 255)

        answer['plot'] = Image.fromarray(yolo_answer.plot(conf=False, line_width=1, font_size=10))

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

        self.start_frame = 0

    def set_formats(self):
        return ('.mp4', '.cine')

    def _set_saving_rule(self) -> Dict[str, Path]:
        rules = {}

        if self.result_dir is None:
            print(f'result_dir is not set! This prevents video creation and other saving.\nTo prevent this later pass result_dir to the init.')
            return rules

        exp_path = Path(self.result_dir) / self.exp_name
        exp_path.mkdir(parents=True, exist_ok=True)
        print(f'Results will be here: {exp_path.as_posix()}')

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
        self.start_frame = start_frame
        metadata_dict['start_frame'] = start_frame
        metadata_dict['stream'] = frame_stream[start_frame:end_frame]
        metadata_dict['stream_len'] = len(metadata_dict['stream'])
        metadata_dict['end_frame'] = start_frame + len(metadata_dict['stream'])

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
        try:
            self.form_result_xlsx(states)
        except Exception as e:
            print(f'Error while forming the result xlsx file: {e}')

        return states

    def form_result_xlsx(self, states: Dict[str, Any]) -> None:
        """Form the result xlsx file."""
        if self.result_dir is None:
            return None

        exp_path = self.result_dir / self.exp_name
        exp_path.mkdir(parents=True, exist_ok=True)

        df_diam_results = pd.DataFrame()
        df_rupture_basic_results = pd.DataFrame()
        df_rupture_track_results = pd.DataFrame()
        df_rupture_all_result = pd.DataFrame()
        rupture_life = {}
        death_successors = {}

        for frame_idx, frame_results in states['result'].items():
            real_frame_idx = frame_idx + states['start_frame']
            for name, stat in frame_results.items():
                if name == 'droplet_stat':
                    df_diam_results = pd.concat((df_diam_results, pd.DataFrame.from_dict({real_frame_idx: stat}, orient='index')))
                elif name == 'ruptures_stat':
                    single_frame_df = pd.DataFrame.from_dict(
                        {
                            real_frame_idx: {
                                'num_ruptures': len(stat),
                                'ruptures_total_area': frame_results['full_mask_bool'][..., 2].sum(),
                            },
                        },
                        orient='index',
                    )
                    df_rupture_basic_results = pd.concat((df_rupture_basic_results, single_frame_df))
                    rupture_area_single_stat = pd.DataFrame.from_dict(
                        {
                            real_frame_idx: {rupt_idx: single_rupture['px_area'] for rupt_idx, single_rupture in stat.items()},
                        },
                        orient='index',
                    )
                    df_rupture_track_results = pd.concat((df_rupture_track_results, rupture_area_single_stat))
                    df_rupture_all_result = pd.concat((df_rupture_all_result, pd.DataFrame.from_dict({real_frame_idx: stat}, orient='index')))

                    for rupt_idx, single_rupture_stat in stat.items():
                        if rupt_idx in rupture_life:
                            rupture_life[rupt_idx][frame_idx] = single_rupture_stat
                        else:
                            rupture_life[rupt_idx] = {frame_idx: single_rupture_stat}

        for rupt_idx, rupture_stat in rupture_life.items():
            dead_frame = max(list(rupture_stat.keys()))
            if rupture_stat[dead_frame].get('nearest_rupture_iou', 0) > 0.1:
                death_successors[rupt_idx] = {'successor': rupture_stat[dead_frame]['nearest_rupture_id'], 'frame': dead_frame + self.start_frame}
            elif rupture_stat[dead_frame].get('nearest_rupture_id', None) is not None:
                nearest_id = rupture_stat[dead_frame].get('nearest_rupture_id')
                if dead_frame + 1 in rupture_life[nearest_id] and dead_frame in rupture_life[nearest_id]:
                    has_grown = rupture_life[nearest_id][dead_frame + 1]['px_area'] > 1.2 * rupture_life[nearest_id][dead_frame]['px_area']
                    if has_grown:
                        death_successors[rupt_idx] = {'successor': nearest_id, 'frame': dead_frame + self.start_frame}
    
        with pd.ExcelWriter(exp_path / f'{self.exp_name}_result_stat.xlsx') as writer:
            df_diam_results.to_excel(writer, sheet_name='Droplet_Diam', float_format="%.3f")
            df_rupture_basic_results.to_excel(writer, sheet_name='Ruptures_Total', float_format="%.3f")
            df_rupture_track_results.to_excel(writer, sheet_name='Ruptures_Track_Area', float_format="%.3f")
            df_rupture_all_result.to_excel(writer, sheet_name='Ruptures_All', float_format="%.3f")
            pd.DataFrame.from_dict(death_successors, orient='index').to_excel(writer, sheet_name='Ruptures_Death', float_format="%.3f")

    def _blend_image_for_video(self, state: Dict[str, Any], frame_idx: int) -> bool:
        if self.video_writer is None:
            return False

        # blended_frame = Image.blend(state['image'].convert('RGB'), state['full_mask'].convert('RGB'), 0.2)
        ImageDraw.Draw(state['plot']).text(
            (0, 0),
            f'Frame {frame_idx + self.start_frame}',
            (255, 255, 255),
        )

        # Save the image as a frame of a video
        self.video_writer.write(
            np.asarray(state['plot'])[..., ::-1],
        )
        return True
