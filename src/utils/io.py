from pathlib import Path
from typing import Union


def str_to_path(path: Union[Path, str], check_exist: bool = False) -> Path:
    """
    Convert string to Posixpath.
    :param path: A str or posixpath path.
    :param check_exist: Whereas check the existence of the file.
    :return: posixpath.
    """
    if isinstance(path, str):
        path = Path(path)

    if check_exist and not path.exists():
        raise FileExistsError(f'There is no such {path.as_posix()}.')

    return path
