import pathlib

import cv2
import httpx
import numpy as np

import research_utilities.common as _common


def get_demo_image() -> pathlib.Path:
    logger = _common.get_logger()

    file_path = _common.GlobalSettings.CACHE_DIR / 'assets' / 'kodim08.png'

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

        url = 'https://r0k.us/graphics/kodak/kodak/kodim08.png'
        url_data = httpx.get(url).read()

        if url_data is None:
            raise ValueError(f'Failed to download image from {url}')
        if not url_data:
            raise ValueError(f'Empty response from {url}')

        with open(file_path, 'wb') as f:
            f.write(url_data)
            logger.info(f'Downloaded an demo image to {file_path}')

    return file_path


def get_checkerboard_image() -> pathlib.Path:
    logger = _common.get_logger()

    file_path = _common.GlobalSettings.CACHE_DIR / 'assets' / 'checkerboard.png'

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

        canvas = np.zeros((512, 512, 3), dtype=np.uint8)

        canvas[0:256, 0:256] = [255, 255, 255]
        canvas[256:512, 256:512] = [255, 255, 255]

        cv2.imwrite(str(file_path), canvas)
        logger.info(f'Created a checkerboard image at {file_path}')

    return file_path
