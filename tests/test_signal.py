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
    logger.info(f'checkerboard_img: {checkerboard_img}')

    img = cv2.imread(checkerboard_img)

    img = torch.from_numpy(img).to(torch.float32).cuda()
    img = img / 255.0
    img = img.unsqueeze(0).permute(0, 3, 1, 2)  # NCHW

    logger.debug(f'img.shape: {img.shape}')
    logger.debug(f'img.dtype: {img.dtype}')
    logger.debug(f'img.min(): {img.min()}, img.max(): {img.max()}')

    psd = _signal.calc_radial_psd_profile(img)

    logger.debug(f'psd.shape: {psd.shape}')
    logger.debug(f'psd.dtype: {psd.dtype}')
    logger.debug(f'psd.min(): {psd.min()}, psd.max(): {psd.max()}')


def test_power_spectral_density_cpu_gpu(checkerboard_img: str, logger: logging.Logger) -> None:
    logger.info(f'checkerboard_img: {checkerboard_img}')

    img = cv2.imread(checkerboard_img)

    img = torch.from_numpy(img).to(torch.float32)
    img = img / 255.0
    img = img.unsqueeze(0).permute(0, 3, 1, 2)  # NCHW

    # cpu
    psd_cpu = _signal.calc_radial_psd_profile(img.detach().cpu(), enable_omp=False)

    # openmp
    psd_omp = _signal.calc_radial_psd_profile(img.detach().cpu(), enable_omp=True)

    # gpu
    psd_gpu = _signal.calc_radial_psd_profile(img.detach().cuda()).cpu()

    logger.debug(f'psd_cpu.shape: {psd_cpu.shape}')
    logger.debug(f'psd_cpu.dtype: {psd_cpu.dtype}')
    logger.debug(f'psd_cpu.min(): {psd_cpu.min()}, psd_cpu.max(): {psd_cpu.max()}')

    logger.debug(f'psd_omp.shape: {psd_omp.shape}')
    logger.debug(f'psd_omp.dtype: {psd_omp.dtype}')
    logger.debug(f'psd_omp.min(): {psd_omp.min()}, psd_omp.max(): {psd_omp.max()}')

    logger.debug(f'psd_gpu.shape: {psd_gpu.shape}')
    logger.debug(f'psd_gpu.dtype: {psd_gpu.dtype}')
    logger.debug(f'psd_gpu.min(): {psd_gpu.min()}, psd_gpu.max(): {psd_gpu.max()}')

    tolerance = 1e-3

    assert torch.isnan(psd_cpu).sum() == 0, "NaN values found in CPU result"
    assert torch.isnan(psd_omp).sum() == 0, "NaN values found in OMP result"
    assert torch.isnan(psd_gpu).sum() == 0, "NaN values found in GPU result"

    def relative_error(a, b):
        return torch.abs(a - b) / (torch.abs(b) + 1e-8)

    relerr_cpu_omp = relative_error(psd_cpu, psd_omp)
    relerr_omp_gpu = relative_error(psd_omp, psd_gpu)
    relerr_gpu_cpu = relative_error(psd_gpu, psd_cpu)

    assert torch.all(relerr_cpu_omp < tolerance), f"CPU and OMP results differ by more than {tolerance}: {relerr_cpu_omp.max()}"
    assert torch.all(relerr_omp_gpu < tolerance), f"OMP and GPU results differ by more than {tolerance}: {relerr_omp_gpu.max()}"
    assert torch.all(relerr_gpu_cpu < tolerance), f"GPU and CPU results differ by more than {tolerance}: {relerr_gpu_cpu.max()}"
