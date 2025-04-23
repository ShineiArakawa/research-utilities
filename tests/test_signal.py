import logging

import cv2
import torch

import research_utilities.resampling as _resampling
import research_utilities.signal as _signal


def test_fft_2d(test_image: str, logger: logging.Logger) -> None:
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    logger.debug(f'img.shape: {img.shape}')

    _signal.fft_2d(img, 'fft.png')


def test_bilinear_interp(test_image: str, logger: logging.Logger) -> None:
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    logger.debug(f'img.shape: {img.shape}')

    img = torch.from_numpy(img).to(torch.float32).cuda()

    img = _resampling.resample(img, 2.0)
    logger.debug(f'img.shape: {img.shape}')


def test_power_spectral_density(checkerboard_img: str, logger: logging.Logger) -> None:
    img = cv2.imread(checkerboard_img)

    img = torch.from_numpy(img).to(torch.float32).cuda()
    img = img / 255.0
    img = img.unsqueeze(0).permute(0, 3, 1, 2)  # NCHW

    logger.debug(f'img.shape: {img.shape}')
    logger.debug(f'img.dtype: {img.dtype}')
    logger.debug(f'img.min(): {img.min()}, img.max(): {img.max()}')

    psd = _signal.calc_power_spectral_density(img)

    logger.debug(f'psd.shape: {psd.shape}')
    logger.debug(f'psd.dtype: {psd.dtype}')
    logger.debug(f'psd.min(): {psd.min()}, psd.max(): {psd.max()}')
