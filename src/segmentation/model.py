from pathlib import Path
from typing import Union

import torch
from segmentation_models_pytorch import DeepLabV3

from src.utils import str_to_path, get_config_from_path


def load_config_and_model(model_root: Union[Path, str]) -> DeepLabV3:
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


def setup_segmentation_model(config, load_ckpt: bool = False) -> DeepLabV3:
    """
    Setup model for facade segmentaion task.
    :param config: Easydict config with model state.
    :param load_ckpt: A path to the ckpt file to load in.
    :return: Pytorch segmentation model.
    """
    model = DeepLabV3(
        encoder_name=config.encoder_name,
        encoder_weights=None if load_ckpt else 'imagenet',
        classes=config.segm_num_classes,
    )

    if load_ckpt:
        state_dict = torch.load(config.ckpt_path, map_location='cpu')
        if 'optimizer' in state_dict.keys():
            model.load_state_dict(state_dict['model'])
        else:
            model.load_state_dict(state_dict)

    model.to(config.device)
    model.eval()

    model.config = config

    return model
