import argparse
import importlib
from pathlib import Path
from typing import Union

from src.utils.io import str_to_path


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help='Path to a config file.',
    )
    args = parser.parse_args()
    path_to_config = Path(args.config)

    return get_config_from_path(path_to_config)


def get_config_from_path(path_to_config: Union[Path, str]):
    """
    Read config and return easydict object.

    :param path_to_config: posixpath
    :return: easydict config
    """
    path_to_config = str_to_path(path_to_config, True)

    # Add default config as template
    config = importlib.import_module("configs.default")
    cfg = config.config

    # Update default config with provided one
    temp_config_name = path_to_config.stem
    config = importlib.import_module(f"configs.{temp_config_name}")
    cfg.update(config.config)

    cfg.config_path = path_to_config

    return cfg
