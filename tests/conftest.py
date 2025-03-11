import logging
import tempfile

import pytest
import requests


@pytest.fixture(scope='module')
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope='module')
def lenna():
    url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
    url_data = requests.get(url).content

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
        f.write(url_data)
        yield f.name


@pytest.fixture(scope='module')
def logger() -> logging.Logger:
    return logging.getLogger(__name__)
