import logging
import tempfile

import pytest

import research_utilities.demo_imgs as _demo_imgs


@pytest.fixture(scope='module')
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope='module')
def test_image() -> str:
    return str(_demo_imgs.get_demo_image())


@pytest.fixture(scope='module')
def checkerboard_img():
    return str(_demo_imgs.get_checkerboard_image())


@pytest.fixture(scope='module')
def logger() -> logging.Logger:
    return logging.getLogger(__name__)
