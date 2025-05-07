"""Demo images for testing and debugging on the fly.
"""

import pathlib

import cv2
import httpx
import numpy as np

import research_utilities.common as _common


def get_demo_image(idx: int = 8) -> pathlib.Path:
    """
    Download a demo image from the internet and save it to the cache directory.
    The images are from the Kodak dataset (URL: https://r0k.us/graphics/kodak/).

    Parameters
    ----------
    idx : int
        The index of the image to download. Default is 8.
        The images are named 'kodim01.png', 'kodim02.png', etc.
        The index should be between 1 and 24.

    Returns
    -------
    pathlib.Path
        The path to the downloaded image.
    """

    logger = _common.get_logger()

    # Check if the index is valid
    if idx < 1 or idx > 24:
        raise ValueError(f'Invalid index {idx}. Must be between 1 and 24.')

    file_name = f'kodim{idx:02d}.png'
    file_path = _common.GlobalSettings.CACHE_DIR / 'assets' / file_name

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

        url = f'https://r0k.us/graphics/kodak/kodak/{file_name}'
        url_data = httpx.get(url).read()

        if url_data is None:
            raise ValueError(f'Failed to download image from {url}')
        if not url_data:
            raise ValueError(f'Empty response from {url}')

        with open(file_path, 'wb') as f:
            f.write(url_data)
            logger.info(f'Downloaded an demo image to {file_path}')

    return file_path


def get_checkerboard_image(size: tuple[int, int] = (512, 512)) -> pathlib.Path:
    r"""
    Create a checkerboard image and save it to the cache directory.

    Like ...
    + ------ + ------ +
    |        | ++++++ |
    |        | ++++++ |
    |        | ++++++ |
    + ------ + ------ +
    | ++++++ |        |
    | ++++++ |        |
    | ++++++ |        |
    + ------ + ------ +

    Parameters
    ----------
    size : tuple[int, int]
        The size of the checkerboard image. Default is (512, 512).
        The image will be divided into 4 quadrants, each with a size of (size[0] // 2, size[1] // 2).

    Returns
    -------
    pathlib.Path
        The path to the checkerboard image.
    """

    logger = _common.get_logger()

    file_path = _common.GlobalSettings.CACHE_DIR / 'assets' / 'checkerboard.png'

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

        canvas = np.zeros((*size, 3), dtype=np.uint8)

        half_size = (size[0] // 2, size[1] // 2)

        canvas[0:half_size[0], 0:half_size[1]] = [255, 255, 255]
        canvas[half_size[0]:, half_size[1]:] = [255, 255, 255]

        cv2.imwrite(str(file_path), canvas)
        logger.info(f'Created a checkerboard image at {file_path}')

    return file_path
