from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from easydict import EasyDict
from PIL import Image
from tqdm import tqdm

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

        self.model = self._setup_model()
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

    def _setup_model(self) -> Callable:
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
    def __init__(
        self,
        config: Optional[EasyDict],
        result_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if config:
            self.config = config
        else:
            self.config = get_config_from_path('./configs/default.py')

        self.config.device = 'cpu' if not torch.cuda.is_available() else self.config.device
        self.device = self.config.device

        self.model_pipeline = self.setup_model_pipeline()
        self.supported_formats = self.set_formats()

        self.result_dir = Path(result_dir) if isinstance(result_dir, str) else result_dir

        # Should be set later by the child class
        self.video_writer = None

    def _set_saving_rule(self) -> Dict[str, Path]:
        """Set saving rule for the results of the pipeline.

        The rule is a dictionary where the keys are the names of the outputs 
        and the values are the paths to save them.

        Returns:
            Dict[str, Path]: Saving rule for the results
        """
        if self.result_dir is None:
            return {}

        return {
            'image': self.result_dir / f'{self.exp_name}' / 'original_frames',
        }

    @staticmethod
    def _setup_result_dirs(saving_rule: Dict[str, Dict[str, Union[Path, Callable]]]) -> None:
        for result_rule in saving_rule.values():
            if 'path' in result_rule:
                result_rule['path'].mkdir(parents=True, exist_ok=True)

    def setup_model_pipeline(self) -> List[Callable]:
        raise NotImplementedError

    def set_formats(self):
        raise NotImplementedError

    def process(self, input_file_raw: Union[Path, str]):
        """Process input file and return the result of the model pipeline.

        Args:
            input_file (Union[Path, str]): Path to the input file
        """
        input_file: Path = self.check_file(input_file_raw)
        states = self._open_file(input_file)

        self.exp_name = input_file.stem

        self.saving_rule = self._set_saving_rule()
        self._setup_result_dirs(self.saving_rule)

        states['result'] = {}

        if 'stream' in states:
            states = self._process_stream(states)
        elif 'image' in states:
            states = self._process_image(states)
        else:
            raise ValueError('No image or stream found in the input state!')

        return self._get_final_result(states)

    def _process_stream(self, states):
        for frame_idx, single_frame in tqdm(enumerate(states['stream']), total=states['stream_len']):
            single_frame_state = self._preprocess_input(single_frame)
            states['result'][frame_idx] = {'image': single_frame_state['image']}

            for pipe in self.model_pipeline:
                single_frame_state = pipe(single_frame_state)
                states['result'][frame_idx].update(single_frame_state)

            self._save_single_frame_result(states['result'][frame_idx], frame_idx)

        # Remove the stream from the states
        del states['stream']

        return states

    def _process_image(self, states):
        for pipe in self.model_pipeline:
            states.update(pipe(states))
        # TODO: Check if the result
        return self._save_single_frame_result(states['result'][0], 0)

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
            input_file = Path(input_file).resolve()
        elif not isinstance(input_file, Path):
            raise ValueError(f'Invalid input file: {input_file}')

        if not input_file.exists():
            raise FileNotFoundError(f'There is no such file as {input_file.as_posix()}')
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

    def _save_single_frame_result(self, state: Dict[str, Any], state_idx: int) -> Dict[str, Any]:
        """Save the processing result of a single frame.

        Args:
            state (Dict[str, Any]): A single frame state from the model pipeline.

        Returns:
            Dict[str, Any]: _description_
        """
        for result_type, save_method in self.saving_rule.items():
            if isinstance(save_method, dict) and 'path' in save_method and result_type in state:
                save_path = save_method['path'] / f'{self.exp_name}_{state_idx}_{result_type}.png'
                try:
                    is_saved = self._file_saver(save_path, state[result_type])
                except Exception as e:
                    print(f'Failed to save {result_type} to {save_path.as_posix()} with error:\n{e}')
                else:
                    state[f'{result_type}_path'] = save_path.as_posix() if is_saved else None
    
            elif isinstance(save_method, dict) and 'save_func' in save_method:
                # Use custom save function
                try:
                    is_saved = save_method['save_func'](state, state_idx)
                except Exception as e:
                    print(f'Failed to save {result_type} using {save_method["save_func"].__name__} with error:\n{e}')
                else:
                    if not is_saved:
                        print(f'Failed to save {result_type} using {save_method["save_func"].__name__} for id {state_idx}!')
        return state
    
    def _file_saver(
        self,
        file_path: Path,
        obj: Image.Image,
    ) -> bool:
        if file_path.suffix in image_extensions:
            # Save the image as PIL image
            obj.save(file_path)
            return True

        return False
