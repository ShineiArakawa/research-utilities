"""Provide signal processing tools including 1D, 2D FFT, Radial PSD profile,
"""

import functools
import pathlib
import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import typing_extensions

import research_utilities.cmap as _cmap
import research_utilities.common as _common
import research_utilities.torch_util as _torch_util

ArrayLike = typing.TypeVar('ArrayLike', np.ndarray, torch.Tensor)


@functools.lru_cache()
def _get_cpp_module(is_cuda: bool = True, with_omp: bool = True) -> typing.Any:
    ext_loader = _torch_util.get_extension_loader()

    name = 'signal'
    sources = [
        'signal.cpp',
    ]

    if is_cuda:
        sources += [
            'signal.cu',
            'resampling.cu',
        ]
        name += '_cuda'

    module = ext_loader.load(
        name=name,
        sources=sources,
        debug=_common.GlobalSettings.DEBUG_MODE,
        with_omp=with_omp,
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
    img: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
    if isinstance(img, np.ndarray):
        return _fft_2d_np(img)
    elif isinstance(img, torch.Tensor):
        return _fft_2d_torch(img)
    else:
        raise TypeError(f'Unsupported type: {type(img)}')


def _fft_2d_np(
    img: np.ndarray,
    is_density: bool = False,
    is_db_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fft_img = np.fft.fft2(img)
    fft_img_shifted = np.fft.fftshift(fft_img)
    mag_power_spectrum = np.abs(fft_img_shifted) ** 2

    if is_density:
        mag_power_spectrum /= (img.shape[-2] * img.shape[-1])

    if is_db_scale:
        mag_power_spectrum = 20.0 * np.log10(mag_power_spectrum + 1e-10)

    return fft_img, mag_power_spectrum


def _fft_2d_torch(
    img: torch.Tensor,
    is_density: bool = False,
    is_db_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    fft_img = torch.fft.fft2(img)
    fft_img_shifted = torch.fft.fftshift(fft_img)
    mag_power_spectrum = torch.abs(fft_img_shifted) ** 2

    if is_density:
        mag_power_spectrum /= (img.shape[-2] * img.shape[-1])

    if is_db_scale:
        mag_power_spectrum = 20.0 * torch.log10(mag_power_spectrum + 1e-10)

    return fft_img, mag_power_spectrum


def plot_spectrum_2d(
    img: np.ndarray,
    file_path: str,
    color_map_type: str = 'viridis',
    is_density: bool = True,
    is_db_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D FFT of an image and plot the power spectrum.

    Parameters
    ----------
    img : np.ndarray
        The input image (shape: [height, width]).
    file_path : str
        The file path to save the plot.
    color_map_type : str, optional
        The color map to be applied to the plot.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The 2D FFT and the power spectrum of the image.
    """

    assert img.ndim >= 2
    assert img.ndim == 2 or (img.ndim == 3 and img.shape[0] == 1)

    # Compute the 2D FFT
    fft_img, spectrum = _fft_2d_np(img, is_density, is_db_scale)

    spectrum = spectrum.astype(np.float32)

    path = pathlib.Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    img = _cmap.apply_color_map(
        img=spectrum,
        color_map_type=color_map_type
    )  # img is returned in BGR format and float32 dtype

    # To [0, 1]
    min_val = img.min()
    max_val = img.max()

    img = (img - min_val) / (max_val - min_val)

    # To [0, 255] and uint8
    img = (img * 255).astype(np.uint8)

    cv2.imwrite(file_path, img)

    return fft_img, spectrum


def compute_psd(
    img: ArrayLike,
    is_db_scale: bool = False,
    beta: float | None = None,
    interpolation: int = 4,
) -> ArrayLike:
    """
    Compute the power spectral density of an image.

    Parameters
    ----------
    img : ArrayLike (np.ndarray or torch.Tensor)
        Input image to be processed.
        The input image can have 2, 3, or 4 dimensions.
        - 2D image: [height, width]
        - 3D image: [channels, height, width]
        - 4D image: [batch, channels, height, width]
    is_db_scale : bool, optional
        Whether to convert the power spectral density to dB scale, by default False
    beta : float, optional
        The beta parameter for the Kaiser window. If None, no windowing is applied.
    interpolation : int, optional
        The interpolation factor for the windowing, by default 4
        This is only used when `beta` is specified.
        The input image will be padded to `interpolation * img.shape[-2:]` before applying the FFT.

    Returns
    -------
    ArrayLike (np.ndarray or torch.Tensor)
        The power spectral density of the image. The output image will have the shape `[batch, channel, height, width]`.
        If `is_db_scale` is True, the output will be in dB scale.
    """

    # ----------------------------------------------------------------------------------------------------
    # If the input is a numpy array, convert it to a torch tensor
    is_np = isinstance(img, np.ndarray)
    if is_np:
        # Convert to torch.Tensor
        img = torch.from_numpy(img).float()

    # Check the input
    if img.ndim == 2:
        # Add the batch and channel dimensions
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        # Add the batch dimension
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError(f'Input image must have 2, 3, or 4 dimensions, but got {img.ndim}.')

    img_h, img_w = img.shape[-2:]

    # ----------------------------------------------------------------------------------------------------
    # Windowing

    if beta is not None and beta >= 0.0:
        assert interpolation > 1, 'Interpolation must be greater than 1 when beta is specified.'

        # Prepare kaiser window
        short_side = max(img_h, img_w)

        window = torch.kaiser_window(short_side, periodic=False, beta=beta, device=img.device, dtype=img.dtype)
        window *= window.square().sum().rsqrt()
        window = window.ger(window)  # [short_side, short_side]
        window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, short_side, short_side]

        if short_side != img_h or short_side != img_w:
            padding_h = (short_side - img_h) // 2
            padding_w = (short_side - img_w) // 2
            window = torch.nn.functional.pad(window, (padding_w, padding_w, padding_h, padding_h))  # [left, right, top, bottom]

        assert window.shape[-2] == img_h
        assert window.shape[-1] == img_w

        # Apply window
        padding_h = (img_h * interpolation - img_h)
        padding_w = (img_w * interpolation - img_w)

        img = torch.nn.functional.pad(img * window, (0, padding_w, 0, padding_h))

    # ----------------------------------------------------------------------------------------------------
    # Apply FFT

    spectrum = torch.fft.fftn(img, dim=(-2, -1)).abs().square()  # [batch, channel, height, width]
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))  # Shift the zero frequency component to the center

    spectrum = spectrum / (img.shape[-2] * img.shape[-1])  # Normalize

    if is_db_scale:
        spectrum = 20.0 * torch.log10(spectrum + 1e-10)

    # ----------------------------------------------------------------------------------------------------
    # Convert back to numpy array if the input was a numpy array
    if is_np:
        # Convert back to numpy array
        spectrum = spectrum.cpu().numpy()

    # ----------------------------------------------------------------------------------------------------
    return spectrum


def _calc_radial_psd_profile_impl(
    psd: torch.Tensor,
    n_divs: int,
    n_points: int,
    enable_omp: bool = True,
) -> torch.Tensor:
    """Calculate the radial power spectral density.

    Parameters
    ----------
    psd : torch.Tensor
        Input power spectral density. Must be 4D tensor with shape [batch, channel, height, width].
    n_divs : int
        Number of bins for the radial angle.
    n_points : int
        Number of points for the polar coordinate.
    enable_omp : bool, optional
        Whether to enable OpenMP for parallel processing, by default True.
        This only affects the C++ implementation.

    Returns
    -------
    torch.Tensor
        Radial power spectral density profile. The output will have the shape [batch, channel, n_divs, n_points].
    """

    # Convert to channels last format
    psd = psd.permute(0, 2, 3, 1).contiguous()  # [batch, height, width, channel]

    # Compute the power spectral density
    _module = _get_cpp_module(is_cuda=psd.is_cuda, with_omp=enable_omp)

    radial_profile = _module.calc_radial_psd_profile(
        psd,
        n_divs,
        n_points
    )

    # Convert back to channels first format
    radial_profile = radial_profile.permute(0, 3, 1, 2)  # [batch, channel, n_divs, n_points]

    return radial_profile


@typing_extensions.deprecated("This function is deprecated and will be removed in future versions. Use `compute_radial_psd` instead.")
def calc_radial_psd_profile(
    img: torch.Tensor,
    n_divs: int = 180,
    n_points: int = 512,
    enable_omp: bool = True,
) -> torch.Tensor:
    """
    Compute the radial power spectral density profile of an image.

    Parameters
    ----------
    img : torch.Tensor
        Input image to be processed.
        The input image can have 2, 3, or 4 dimensions.
        - 2D image: [height, width]
        - 3D image: [channels, height, width]
        - 4D image: [batch, channels, height, width]
    n_divs : int, optional
        Number of divisions for the radial angle, by default 180
    n_points : int, optional
        Number of points for the radial profile, by default 512
    enable_omp : bool, optional
        Whether to enable OpenMP for parallel processing, by default True
        This only affects the C++ implementation.

    Returns
    -------
    torch.Tensor
        Radial power spectral density profile. The output image will have the shape `[batch, channel, n_divs, n_points]`.

    Raises
    ------
    ValueError
        If the input image has an invalid number of dimensions.
    """

    assert isinstance(img, torch.Tensor), 'The input image must be a torch.Tensor.'

    # Check the input
    if img.ndim == 2:
        # Add the batch and channel dimensions
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        # Add the batch dimension
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError(f'Input image must have 2, 3, or 4 dimensions, but got {img.ndim}.')

    # Apply FFT
    _, psd = _fft_2d_torch(img, is_density=True, is_db_scale=False)  # [batch, channel, height, width]

    return _calc_radial_psd_profile_impl(
        psd=psd,
        n_divs=n_divs,
        n_points=n_points,
        enable_omp=enable_omp
    )


def compute_radial_psd(
    img: ArrayLike,
    n_divs: int = 180,
    n_points: int = 512,
) -> ArrayLike:
    """
    Compute the radial power spectral density of images.

    Parameters
    ----------
    img : ArrayLike (np.ndarray or torch.Tensor)
        Input image to be processed.
        The input image can have 2, 3, or 4 dimensions.
        - 2D image: [height, width]
        - 3D image: [channels, height, width]
        - 4D image: [batch, channels, height, width]
    n_divs : int, optional
        Number of divisions for the radial angle, by default 180
    n_points : int, optional
        Number of points for the radial profile, by default 512

    Returns
    -------
    ArrayLike (np.ndarray or torch.Tensor)
        Radial power spectral density profile. The output image will have the shape `[batch, channel, n_divs, n_points]`.

    Raises
    ------
    ValueError
        If the input image has an invalid number of dimensions.
    """

    # ----------------------------------------------------------------------------------------------------
    # If the input is a numpy array, convert it to a torch tensor
    is_np = isinstance(img, np.ndarray)
    if is_np:
        # Convert to torch.Tensor
        img = torch.from_numpy(img).float()

    # Check the input
    if img.ndim == 2:
        # Add the batch and channel dimensions
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        # Add the batch dimension
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError(f'Input image must have 2, 3, or 4 dimensions, but got {img.ndim}.')

    # ----------------------------------------------------------------------------------------------------
    # Execute FFT with Kaiser windowing
    psd = compute_psd(img, is_db_scale=False, beta=8.0, interpolation=4)

    # ----------------------------------------------------------------------------------------------------
    # Compute the radial power spectral density
    radial_psd = _calc_radial_psd_profile_impl(
        psd=psd,
        n_divs=n_divs,
        n_points=n_points,
        enable_omp=True
    )

    # ----------------------------------------------------------------------------------------------------
    if is_np:
        # Convert back to numpy array
        radial_psd = radial_psd.cpu().numpy()

    return radial_psd


def radial_freq(
    img_size: int,
    n_points: int,
    dtype: typing.Type = np.float32,
) -> np.ndarray:
    """Generate radial frequency values for a given image size and number of points.

    Parameters
    ----------
    img_size : int
        The size of the image (assumed to be square).
    n_points : int
        The number of points in the radial frequency.
    dtype : type, optional
        The data type of the output array, by default np.float32.

    Returns
    -------
    np.ndarray
        The radial frequency values for the real parts.
    """

    assert isinstance(img_size, int) and img_size > 0, 'img_size must be a positive integer.'
    assert isinstance(n_points, int) and n_points > 0, 'n_points must be a positive integer.'

    max_index = (img_size - 1) // 2

    # NOTE: Reference for the frequency calculation:
    #     'numpy.fft.fftfreq' in NumPy's API reference (URL: https://numpy.org/doc/2.2/reference/generated/numpy.fft.fftfreq.html#numpy-fft-fftfreq)
    #     'numpy/fft/_helper.py' (URL: https://github.com/numpy/numpy/blob/e7a123b2d3eca9897843791dd698c1803d9a39c2/numpy/fft/_helper.py#L125-L177)
    #
    # f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    # f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    freq = np.linspace(0.0, float(max_index), n_points, dtype=dtype, endpoint=True) / (float(img_size) * 1.0)  # d = 1.0

    return freq
