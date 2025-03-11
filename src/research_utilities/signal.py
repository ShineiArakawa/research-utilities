import pathlib

import numpy as np


def plot_signal(
    signal: np.ndarray,
    file_path: str,
    title: str | None = None,
    label: str | None = None,
    x_label: str = 'Time',
    y_label: str = 'Amplitude',
    grid_alpha: float = 0.3
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
    grid_alpha : float, optional
        The transparency of the grid.
    """

    # Plot the signal
    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    ax.plot(signal, label=label)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

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
    grid_alpha: float = 0.3
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
    grid_alpha : float, optional
        The transparency of the grid.

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
        import matplotlib.pyplot as plt
        fig = plt.figure(dpi=300)
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
