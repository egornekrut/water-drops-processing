from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from src.fframe.model import FrameClassModel
from src.utils.inference_v2 import BasicModelPipeline


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
