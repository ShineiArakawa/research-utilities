import logging
import tempfile

import conftest
import cv2
import pytest

import research_utilities.signal as _signal


def test_fft_2d(lenna: str, logger: logging.Logger) -> None:
    img = cv2.imread(lenna, cv2.IMREAD_GRAYSCALE)
    logger.debug(f'img.shape: {img.shape}')

    freq, amp = _signal.fft_2d(img, 'fft.png')
