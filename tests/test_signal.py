import logging

import cv2
import torch

import research_utilities.resampling as _resampling
import research_utilities.signal as _signal


def to_cuda(arr: torch.Tensor) -> torch.Tensor:
    """Move tensor to GPU if available."""
    if isinstance(arr, torch.Tensor) and torch.cuda.is_available():
        return arr.cuda()
    return arr


def test_fft_2d(test_image: str, logger: logging.Logger) -> None:
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    logger.debug(f'img.shape: {img.shape}')

    _signal.fft_2d(img)


def _interp_test_impl(
    test_image: str,
    interp_method: _resampling.InterpMethod,
    logger: logging.Logger,
) -> None:
    if not torch.cuda.is_available():
        logger.warning('CUDA is not available. Skipping interpolation test.')
        return

    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    logger.debug(f'img.shape: {img.shape}')

    img = to_cuda(torch.from_numpy(img).to(torch.float32))

    img = _resampling.resample(img, 4.0, interp_method=interp_method)
    logger.debug(f'img.shape: {img.shape}')


def test_nearest_interp(test_image: str, logger: logging.Logger) -> None:
    _interp_test_impl(test_image, _resampling.InterpMethod.NEAREST, logger)


def test_bilinear_interp(test_image: str, logger: logging.Logger) -> None:
    _interp_test_impl(test_image, _resampling.InterpMethod.BILINEAR, logger)


def test_bicubic_interp(test_image: str, logger: logging.Logger) -> None:
    _interp_test_impl(test_image, _resampling.InterpMethod.BICUBIC, logger)


def test_lanczos4_interp(test_image: str, logger: logging.Logger) -> None:
    _interp_test_impl(test_image, _resampling.InterpMethod.LANCZOS4, logger)


def test_power_spectral_density(checkerboard_img: str, logger: logging.Logger) -> None:
    logger.info(f'checkerboard_img: {checkerboard_img}')

    img = cv2.imread(checkerboard_img)

    img = to_cuda(torch.from_numpy(img).to(torch.float32))
    img = img / 255.0
    img = img.unsqueeze(0).permute(0, 3, 1, 2)  # NCHW

    logger.debug(f'img.shape: {img.shape}')
    logger.debug(f'img.dtype: {img.dtype}')
    logger.debug(f'img.min(): {img.min()}, img.max(): {img.max()}')

    psd = _signal.calc_radial_psd_profile(img)

    logger.debug(f'psd.shape: {psd.shape}')
    logger.debug(f'psd.dtype: {psd.dtype}')
    logger.debug(f'psd.min(): {psd.min()}, psd.max(): {psd.max()}')


def test_power_spectral_density_cpu_openmp_gpu(checkerboard_img: str, logger: logging.Logger) -> None:
    logger.info(f'checkerboard_img: {checkerboard_img}')

    is_cuda_available = torch.cuda.is_available()

    img = cv2.imread(checkerboard_img)

    img = torch.from_numpy(img).to(torch.float32)
    img = img / 255.0
    img = img.unsqueeze(0).permute(0, 3, 1, 2)  # NCHW

    # cpu
    psd_cpu = _signal.calc_radial_psd_profile(img.detach().cpu(), enable_omp=False)

    # openmp
    psd_omp = _signal.calc_radial_psd_profile(img.detach().cpu(), enable_omp=True)

    if is_cuda_available:
        # gpu
        psd_gpu = _signal.calc_radial_psd_profile(img.detach().cuda()).cpu()

    logger.debug(f'psd_cpu.shape: {psd_cpu.shape}')
    logger.debug(f'psd_cpu.dtype: {psd_cpu.dtype}')
    logger.debug(f'psd_cpu.min(): {psd_cpu.min()}, psd_cpu.max(): {psd_cpu.max()}')

    logger.debug(f'psd_omp.shape: {psd_omp.shape}')
    logger.debug(f'psd_omp.dtype: {psd_omp.dtype}')
    logger.debug(f'psd_omp.min(): {psd_omp.min()}, psd_omp.max(): {psd_omp.max()}')

    if is_cuda_available:
        logger.debug(f'psd_gpu.shape: {psd_gpu.shape}')
        logger.debug(f'psd_gpu.dtype: {psd_gpu.dtype}')
        logger.debug(f'psd_gpu.min(): {psd_gpu.min()}, psd_gpu.max(): {psd_gpu.max()}')

    tolerance = 1e-3

    assert torch.isnan(psd_cpu).sum() == 0, "NaN values found in CPU result"
    assert torch.isnan(psd_omp).sum() == 0, "NaN values found in OMP result"
    if is_cuda_available:
        assert torch.isnan(psd_gpu).sum() == 0, "NaN values found in GPU result"

    def relative_error(a, b):
        return torch.abs(a - b) / (torch.abs(b) + 1e-12)

    relerr_cpu_omp = relative_error(psd_cpu, psd_omp)
    if is_cuda_available:
        relerr_omp_gpu = relative_error(psd_omp, psd_gpu)
        relerr_gpu_cpu = relative_error(psd_gpu, psd_cpu)

    logger.debug(f'relerr_cpu_omp: (min, max) {relerr_cpu_omp.min()}, {relerr_cpu_omp.max()}')
    if is_cuda_available:
        logger.debug(f'relerr_omp_gpu: (min, max) {relerr_omp_gpu.min()}, {relerr_omp_gpu.max()}')
        logger.debug(f'relerr_gpu_cpu: (min, max) {relerr_gpu_cpu.min()}, {relerr_gpu_cpu.max()}')

    assert torch.all(relerr_cpu_omp < tolerance), f"CPU and OMP results differ by more than {tolerance}: {relerr_cpu_omp.max()}"
    if is_cuda_available:
        assert torch.all(relerr_omp_gpu < tolerance), f"OMP and GPU results differ by more than {tolerance}: {relerr_omp_gpu.mean()}"
        assert torch.all(relerr_gpu_cpu < tolerance), f"GPU and CPU results differ by more than {tolerance}: {relerr_gpu_cpu.mean()}"
