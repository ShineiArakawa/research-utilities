import logging
import pathlib
import tempfile

import cv2
import numpy as np
import pytest
import requests


@pytest.fixture(scope='module')
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope='module')
def test_image():
    url = 'https://r0k.us/graphics/kodak/kodak/kodim08.png'
    url_data = requests.get(url).content

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
        f.write(url_data)
        yield f.name


@pytest.fixture(scope='module')
def checkerboard_img():
    file_path = pathlib.Path(__file__).parent / 'assets' / 'checkerboard.png'
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)

        canvas[0:256, 0:256] = [255, 255, 255]
        canvas[256:512, 256:512] = [255, 255, 255]

        cv2.imwrite(str(file_path), canvas)

    return str(file_path.resolve())


@pytest.fixture(scope='module')
def logger() -> logging.Logger:
    return logging.getLogger(__name__)
