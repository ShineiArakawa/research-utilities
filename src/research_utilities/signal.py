import functools
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import research_utilities.apply_color_map as _cm
import research_utilities.common as _common
import research_utilities.torch_util as _torch_util

_logger = _common.get_logger()


@functools.lru_cache(maxsize=None)
def _get_cpp_module():
    ext_loader = _torch_util.get_extension_loader()

    module = ext_loader.load(
        name='signal',
        sources=[
            'signal.cpp',
            'signal.cu',
            'resampling.cu',
        ],
        debug=_common.GlobalSettings.DEBUG_MODE
    )

    return module


def plot_signal(
    signal: np.ndarray,
    file_path: str,
    title: str | None = None,
    label: str | None = None,
    x_label: str = 'Time',
    y_label: str = 'Amplitude',
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    grid_alpha: float = 0.3,
    dpi: int = 200
) -> None:
    """
    Plot a signal.

    Parameters
    ----------
    signal : np.ndarray
        The input signal (shape: [n_points,]).
    file_path : str
        The file path to save the plot.
    title : str, optional
        The title of the plot.
    label : str, optional
        The label of the plot.
    x_label : str, optional
        The x-axis label.
    y_label : str, optional
        The y-axis label.
    x_min : float, optional
        The minimum value of the x-axis.
    x_max : float, optional
        The maximum value of the x-axis.
    y_min : float, optional
        The minimum value of the y-axis.
    y_max : float, optional
        The maximum value of the y-axis.
    grid_alpha : float, optional
        The transparency of the grid.
    dpi : int, optional
        The resolution of the plot.
    """

    # Plot the signal
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)

    ax.plot(signal, label=label)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xlim(left=x_min, right=x_max)
    ax.set_ylim(bottom=y_min, top=y_max)

    if label is not None:
        ax.legend()

    ax.grid(alpha=grid_alpha)

    path = pathlib.Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(path)

    plt.close(fig)
    plt.clf()
    del fig


def fft_1d(
    signal: np.ndarray,
    sampling_rate: float | None = None,
    dt: float | None = None,
    is_db_scale: bool = False,
    file_path: str | None = None,
    title: str | None = None,
    label: str | None = None,
    x_label: str = 'Frequency [Hz]',
    y_label: str = 'Amplitude',
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    grid_alpha: float = 0.3,
    dpi: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 1D Fast Fourier Transform of a signal.

    Parameters
    ----------
    signal : np.ndarray
        The input signal (shape: [n_points,]).
    sampling_rate : float, optional
        The sampling rate of the signal.
    dt : float, optional
        The time interval between each sample.
        `sampling_rate` and `dt` cannot be both None.
    is_db_scale : bool, optional
        Whether to convert the amplitude to dB scale.
    file_path : str, optional
        The file path to save the plot.
    title : str, optional
        The title of the plot.
    label : str, optional
        The label of the plot.
    x_label : str, optional
        The x-axis label.
    y_label : str, optional
        The y-axis label.
    x_min : float, optional
        The minimum value of the x-axis.
    x_max : float, optional
        The maximum value of the x-axis.
    y_min : float, optional
        The minimum value of the y-axis.
    y_max : float, optional
        The maximum value of the y-axis.
    grid_alpha : float, optional
        The transparency of the grid.
    dpi : int, optional
        The resolution of the plot.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The frequency and amplitude of the signal.
    """

    assert signal.ndim == 1
    assert sampling_rate is not None or dt is not None

    # Compute the dt
    if dt is None:
        dt = 1 / sampling_rate

    n_points = len(signal)

    fourier_coeff = np.fft.fft(signal)
    freq = np.fft.fftfreq(n_points, d=dt)

    # Compute the amplitude
    amp = np.abs(fourier_coeff / (n_points / 2))
    if is_db_scale:
        amp = 20 * np.log10(amp)

    if file_path is not None:
        # Plot the FFT
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111)

        ax.plot(
            freq[1:int(n_points / 2)],
            amp[1:int(n_points / 2)],
            label=label
        )

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel(x_label)
        ax.set_ylabel(f'{y_label} [dB]' if is_db_scale else y_label)

        ax.set_xlim(left=x_min, right=x_max)
        ax.set_ylim(bottom=y_min, top=y_max)

        if label is not None:
            ax.legend()

        ax.grid(alpha=grid_alpha)

        path = pathlib.Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        fig.savefig(path)

        plt.close(fig)
        plt.clf()
        del fig

    return freq, amp


def _fft_2d(
    img: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    if isinstance(img, np.ndarray):
        return _fft_2d_np(img)
    elif isinstance(img, torch.Tensor):
        return _fft_2d_torch(img)
    else:
        raise TypeError(f'Unsupported type: {type(img)}')


def _fft_2d_np(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fft_img = np.fft.fft2(img)
    _logger.debug(f'fft_img.shape: {fft_img.shape}')

    fft_img_shifted = np.fft.fftshift(fft_img)
    _logger.debug(f'fft_img_shifted.shape: {fft_img_shifted.shape}')

    mag_spectrum = np.abs(fft_img_shifted)
    _logger.debug(f'mag_spectrum.shape: {mag_spectrum.shape}')

    mag_power_spectrum = np.abs(fft_img_shifted) ** 2
    _logger.debug(f'mag_power_spectrum.shape: {mag_power_spectrum.shape}')

    return fft_img, mag_spectrum, mag_power_spectrum


def _fft_2d_torch(
    img: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fft_img = torch.fft.fft2(img)
    _logger.debug(f'fft_img.shape: {fft_img.shape}')

    fft_img_shifted = torch.fft.fftshift(fft_img)
    _logger.debug(f'fft_img_shifted.shape: {fft_img_shifted.shape}')

    mag_spectrum = torch.abs(fft_img_shifted)
    _logger.debug(f'mag_spectrum.shape: {mag_spectrum.shape}')

    mag_power_spectrum = torch.abs(fft_img_shifted) ** 2
    _logger.debug(f'mag_power_spectrum.shape: {mag_power_spectrum.shape}')

    return fft_img, mag_spectrum, mag_power_spectrum


def fft_2d(
    img: np.ndarray,
    file_path: str | None = None,
    color_map_type: str = 'viridis'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D Fast Fourier Transform of an image.

    Parameters
    ----------
    img : np.ndarray
        The input image (shape: [height, width]).
    file_path : str, optional
        The file path to save the plot.
    color_map_type : str, optional
        The color map to be applied to the plot.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The 2D FFT and the magnitude spectrum of the image.
    """

    _logger.debug(f'img.shape: {img.shape}')

    assert img.ndim >= 2
    assert file_path is None or (img.ndim == 2 or (img.ndim == 3 and img.shape[0] == 1))

    # Compute the 2D FFT
    fft_img, mag_spectrum, _ = _fft_2d(img)
    mag_spectrum = 20.0 * np.log10(mag_spectrum + 1e-10)

    if file_path is not None:
        path = pathlib.Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        mag_spectrum = mag_spectrum.astype(np.float32)

        img = _cm.apply_color_map(
            img=mag_spectrum,
            color_map_type=color_map_type
        )  # img is returned in BGR format and float32 dtype

        # To [0, 1]
        min_val = img.min()
        max_val = img.max()

        _logger.debug(f'min_val: {min_val}, max_val: {max_val}')

        img = (img - min_val) / (max_val - min_val)

        # To [0, 255] and uint8
        img = (img * 255).astype(np.uint8)

        cv2.imwrite(file_path, img)

    return fft_img, mag_spectrum


def calc_radial_psd_profile(
    img: torch.Tensor,
    n_divs: int = 180,
    n_points: int = 512,
) -> torch.Tensor:
    """

    Parameters
    ----------
    img : torch.Tensor
        _description_
    n_divs : int, optional
        _description_, by default 180
    n_points : int, optional
        _description_, by default 512

    Returns
    -------
    torch.Tensor
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    assert isinstance(img, torch.Tensor), 'The input image must be a torch.Tensor.'
    assert img.is_cuda, 'The input image must be on the GPU.'

    # Check the input
    if img.ndim == 2:
        # Add the batch and channel dimensions
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        # Add the batch dimension
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError(f'Input image must have 2, 3, or 4 dimensions, but got {img.ndim}.')

    dir_path = pathlib.Path('debug').resolve()
    dir_path.mkdir(parents=True, exist_ok=True)

    # Apply FFT
    _, _, power_spectrum = _fft_2d_torch(img)  # [batch, channel, height, width]
    psd = power_spectrum / (power_spectrum.shape[2] * power_spectrum.shape[3])  # Normalize by the number of pixels

    # Convert to channels last format
    psd = psd.permute(0, 2, 3, 1).contiguous()  # [batch, height, width, channel]

    # Compute the power spectral density
    _module = _get_cpp_module()

    radial_profile = _module.calc_radial_psd_profile(
        psd,
        n_divs,
        n_points
    )

    # Convert back to channels first format
    radial_profile = radial_profile.permute(0, 3, 1, 2)  # [batch, channel, height, width]

    return radial_profile
