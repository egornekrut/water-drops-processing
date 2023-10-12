from pathlib import Path
from typing import Union

import torch
from segmentation_models_pytorch import UnetPlusPlus, DeepLabV3Plus, Unet
from segmentation_models_pytorch.base import SegmentationModel
from ultralytics import YOLO

from src.utils.io import str_to_path
from src.utils.config import get_config_from_path


def load_config_and_model(model_root: Union[Path, str]) -> SegmentationModel:
    """
    Instantly load the model when config.py and ckpt files are presented in the single location.
    :param model_root: Root directory with config.py and model.ckpt files.
    :return: Loaded pytorch model for facade segmentation.
    """
    model_root = str_to_path(model_root, True)
    config = get_config_from_path(model_root / 'config.py')

    checkpoints = list(model_root.glob('/*.ckpt')).sort()

    if checkpoints is None:
        print(f'There are no "*.ckpt" files in the {model_root.as_posix()}.\n'
              f'Looking for checkpoint in the config file...')
        config.ckpt_path = str_to_path(config.ckpt_path, True)
    else:
        # grab last ckpt
        config.ckpt_path = checkpoints[-1]

    print(f'Loading checkpoint for {config.model_type} from {config.ckpt_path}...')

    return setup_segmentation_model(config, True)


def setup_segmentation_model(config, load_ckpt: bool = False) -> SegmentationModel:
    if config.model_type == 'DL3+':
        model = DeepLabV3Plus(
            encoder_name=config.encoder_name,
            encoder_weights=None if load_ckpt else 'imagenet',
            in_channels=1,
            classes=config.segm_num_classes,
        )
    elif config.model_type == 'unet':
        model = Unet(
            encoder_name=config.encoder_name,
            encoder_weights=None if load_ckpt else 'imagenet',
            in_channels=1,
            classes=config.segm_num_classes,
            decoder_attention_type='scse',
            activation='sigmoid',
        )
    else:
        raise NotImplementedError

    if load_ckpt:
        state_dict = torch.load(config.step_2_ckpt_path, map_location='cpu')
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict['model'])
        else:
            model.load_state_dict(state_dict)

    model.to(config.device)
    model.eval()

    model.config = config

    return model


def setup_detection_model(config) -> YOLO:
    model = YOLO(config.detector_weights)

    return model
