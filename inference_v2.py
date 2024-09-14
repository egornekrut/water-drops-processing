from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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
from src.utils.config import get_config_from_path

image_extensions = ('.png', '.jpeg', '.jpg')
video_extensions = ('.mp4', '.avi', '.mkv', '.cine')


class BasicModelPipeline:
    def __init__(
        self,
        model_config: EasyDict,
        input_keys: List[str],
        output_keys: List[str],
        device: Optional[str] = None,
    ) -> None:
        self.model_config = model_config

        self.model = self._setup_model(model_config)
        self.device = device if torch.cuda.is_available() and device else 'cpu'

        self.model.to(self.device)
        self.model.compile()

        self.input_keys = input_keys
        self.output_keys = output_keys

    @torch.inference_mode()
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        input_data = self._preprocess(input_data)
        output = self._model_inference(input_data)

        return self._postprocess(output)

    def _setup_model(self, model_config: EasyDict) -> Callable:
        raise NotImplementedError

    def _preprocess(self, input_data: Any) -> Any:
        """Preprocess input data before passing to the model.

        Args:
            input_data (Any): Data to be processed by the model

        Returns:
            Any: Processed data
        """
        if len(self.input_keys) == 1 and isinstance(input_data, dict):
            return input_data[self.input_keys[0]]
        else:
            raise NotImplementedError

    def _model_inference(self, input_data: Any) -> Any:
        """Get the output of the model.

        Args:
            input_data (Any): Data to be processed by the model

        Returns:
            Any: Model output
        """
        return self.model(input_data)

    def _postprocess(self, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess model output before returning it to the next pipeline.

        Args:
            model_output (Any): Model output

        Returns:
            Any: Postprocessed data
        """
        answer = {}
        for key in self.output_keys:
            if key not in model_output:
                raise ValueError(f'Output key {key} not found in model output dict!')
            answer[key] = model_output[key]

        return answer


class BasicProcessor:
    def __init__(self, config: Optional[EasyDict]) -> None:
        if config:
            self.config = config
        else:
            self.config = get_config_from_path('./configs/default.py')

        self.config.device = 'cpu' if not torch.cuda.is_available() else self.config.device

        self.model_pipeline = self.setup_model_pipeline()
        self.supported_formats = self.set_formats()

    def setup_model_pipeline(self) -> List[Callable]:
        raise NotImplementedError

    def set_formats(self):
        raise NotImplementedError

    def process(self, input_file: Union[Path, str]):
        """Process input file and return the result of the model pipeline.

        Args:
            input_file (Union[Path, str]): Path to the input file
        """
        input_file = self.check_file(input_file)
        states = self._open_file(input_file)
        states['result'] = {}

        if 'stream' in states:
            for frame_idx, single_frame in tqdm(enumerate(states['stream']), total=states['stream_len']):
                single_frame_state = self._preprocess_input(single_frame)
                states['result'][frame_idx] = {'image': single_frame_state['image']}

                for pipe in self.model_pipeline:
                    single_frame_state = pipe(single_frame_state)
                    states['result'][frame_idx].update(single_frame_state)

            states['stream'].close()
            del states['stream']

        elif 'image' in states:
            for pipe in self.model_pipeline:
                states.update(pipe(states))
        else:
            raise ValueError('No image or stream found in the input state!')

        return self._get_final_result(states)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.process(*args, **kwds)

    def check_file(self, input_file: Union[Path, str]) -> Path:
        """Check if the input file is a valid file and return it as Path.

        Args:
            input_file (Union[Path, str]): A file path or a string representing the file name.

        Returns:
            Path: A valid file path as a Path
        """
        if isinstance(input_file, str):
            input_file = Path(input_file)
        elif not isinstance(input_file, Path):
            raise ValueError(f'Invalid input file: {input_file}')

        if not input_file.exists():
            raise FileNotFoundError
        elif not input_file.suffix in self.supported_formats:
            raise NotImplementedError(f'This format is not supported: {input_file.suffix}.\nTo check supported formats use processor.supported_formats.')

        return input_file

    def _open_file(self, input_file: Path) -> Dict[str, Any]:
        raise NotImplementedError

    def _preprocess_input(self, input_file: Union[Path, str]) -> Dict[str, Any]:
        """Preprocess the input file.

        This method prepares the input file for processing by the model pipeline.

        Args:
            input_file (Union[Path, str]): The input file to be processed.
        """
        raise NotImplementedError

    def _get_final_result(self, states: Dict[str, Any]) -> Any:
        """Get the final result from the processed states.

        Args:
            states (Dict[str, Any]): All the states of the model pipeline.

        Returns:
            Any: Final result of the model pipeline.
        """
        raise NotImplementedError


class YoloDetectorModel(BasicModelPipeline):
    def __init__(
            self,
            model_config,
            device: Optional[str] = None,
    ) -> None:
        super().__init__(model_config, ['image'], ['full_mask'], device)
        self.num_radius = 32

    def _setup_model(self, model_config: EasyDict) -> YOLO:
        return YOLO(model=self.model_config.weights, task='segment')

    def _model_inference(
        self,
        image_pil: Image.Image,
    ) -> Dict[str, Any]:
        yolo_res: Results = self.model.predict(
            image_pil,
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
        # TODO: ДОБАВИТЬ СОВМЕЩЕНИЕ МАСОК, ЕСЛИ НАШЛОСЬ НЕСКОЛЬКО ОБЪЕКТОВ

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
                        }
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


class VideoProcessor(BasicProcessor):
    def __init__(
            self,
            config,
            result_dir: Optional[Union[str, Path]] = None,
            make_video: bool = False,
            save_all_pics: bool = False,
        ) -> None:
        super().__init__(config)
        self.curr_video_format = None
        self.result_dir = Path(result_dir) if isinstance(result_dir, str) else result_dir

        if isinstance(self.result_dir, Path):
            self.result_dir.mkdir(exist_ok=True, parents=True)

        if not self.result_dir and make_video:
            make_video = False
            print(f'result_dir is not set! This prevents video creation. To prevent this later pass result_dir to the class.')

        self.make_video = make_video
        self.save_all_pics = save_all_pics

    def set_formats(self):
        return ('.mp4', '.cine')
    
    def setup_model_pipeline(self):
        return [
            YoloDetectorModel(self.config.yolo_configuration, device=self.config.device),
        ]

    def _open_file(self, input_file: Path) -> Dict[str, Any]:
        metadata_dict = {'input_file': input_file.as_posix()}

        if input_file.suffix == '.mp4':
            self.curr_video_format = 'mp4'
            # frame_stream = cv2.VideoCapture(vid_path.as_posix())
            raise NotImplementedError

        elif input_file.suffix == '.cine':
            self.curr_video_format = 'cine'
            frame_stream = pims.open(input_file.as_posix())
        else:
            raise ValueError(f'Unsupported file type {input_file.stem}')

        metadata_dict['stream'] = frame_stream
        metadata_dict['stream_len'] = len(frame_stream)

        return metadata_dict

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
        if self.result_dir is None:
            return states

        result_dir = self.result_dir / f'{Path(states["input_file"]).stem}'
        result_dir.mkdir(exist_ok=False, parents=False)

        if self.make_video and len(states['result'].keys()) > 1:
            try:
                video_path = self.create_segmentation_video(result_dir, states)
            except:
                print(f'Error during video saving!')
            else:
                states['video_result_path'] = video_path
                print(f'Video with segmentation mask saved to {video_path}')

        if self.save_all_pics:
            try:
                states = self.write_frames_to_disk(result_dir, states)
            except:
                print(f'Error during images saving!')
            else:
                print(f'Images saved to {result_dir.as_posix()}')

        return states

    def create_segmentation_video(self, result_dir: Path, states: Dict[str, Any]) -> str:
        video_path = (result_dir / f'{Path(states["input_file"]).stem}_result.avi').as_posix()

        width, height = states['result'][0]['image'].size
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MPEG'), 10, (width, height))

        for idx, frame_result in states['result'].items():
            video.write(np.asarray(self._blend_image_for_video(frame_result, idx))[..., ::-1])

        cv2.destroyAllWindows()
        video.release()

        return video_path
    
    @staticmethod
    def _blend_image_for_video(state: Dict[str, Image.Image], frame_idx: int) -> Image.Image:
        blended_frame = Image.blend(state['image'].convert('RGB'), state['full_mask'].convert('RGB'), 0.2)
        ImageDraw.Draw(blended_frame).text(
            (0, 0),
            f'Frame {frame_idx}',
            (255, 255, 255),
        )
        return blended_frame

    def write_frames_to_disk(self, result_dir: Path, states: Dict[str, Any]) -> Dict[str, Any]:
        video_name = result_dir.parts[-1]

        original_frames_dir = result_dir / 'original_frames'
        full_masks_dir = result_dir / 'full_masks'

        original_frames_dir.mkdir(exist_ok=False, parents=False)
        full_masks_dir.mkdir(exist_ok=False, parents=False)

        for idx, state in states['result'].items():
            state['image'].save(original_frames_dir / f'{video_name}_original_{idx}.png')
            state['full_mask'].save(full_masks_dir / f'{video_name}_full_mask_{idx}.png')

        states['original_frames_dir'] = original_frames_dir.as_posix()
        states['full_masks_dir'] = full_masks_dir.as_posix()

        return states
