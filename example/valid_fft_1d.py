import matplotlib.pyplot as plt
import numpy as np
import torch


def fft_1d():
    fig = plt.figure()
    axes = fig.subplots(2, 2)

    target_freq_0 = 8.0
    target_freq_1 = 8.5

    sampling_rate = 1000.0
    duration = 1.0

    dt = 1.0 / sampling_rate

    print(f'duration: {duration} [sec], dt: {dt} [sec], sampling_rate: {sampling_rate} [Hz]')

    # ------------------------------------------------------------------------
    x = torch.linspace(0.0, duration, int(duration / dt) + 1)
    y = torch.sin(2.0 * np.pi * target_freq_0 * x)
    axes[0, 0].plot(x, y)
    axes[0, 0].set_title(r'$y = \sin(2\pi \cdot {} \cdot x)$'.format(target_freq_0))
    axes[0, 0].set_xlabel('t [sec]')
    axes[0, 0].set_ylabel('y')

    freq = torch.fft.fftfreq(x.shape[-1], d=dt)
    fourier_coef = torch.fft.fft(y)
    power = torch.abs(fourier_coef) ** 2

    print(freq[1:x.shape[-1] // 2])

    axes[0, 1].plot(freq[1:x.shape[-1] // 2], power[1:x.shape[-1] // 2])
    axes[0, 1].set_title('FFT')
    axes[0, 1].set_xlabel('Frequency [Hz]')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].set_xlim(0, target_freq_0 * 2)

    # ------------------------------------------------------------------------
    x = torch.linspace(0.0, duration, int(duration / dt) + 1)
    y = torch.sin(2.0 * np.pi * target_freq_1 * x)
    axes[1, 0].plot(x, y)
    axes[1, 0].set_title(r'$y = \sin(2\pi \cdot {} \cdot x)$'.format(target_freq_1))
    axes[1, 0].set_xlabel('t [sec]')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_xlim(axes[0, 0].get_xlim())
    axes[1, 0].set_ylim(axes[0, 0].get_ylim())

    freq = torch.fft.fftfreq(x.shape[-1], d=dt)
    fourier_coef = torch.fft.fft(y)
    power = torch.abs(fourier_coef) ** 2

    axes[1, 1].plot(freq[1:x.shape[-1] // 2], power[1:x.shape[-1] // 2])
    axes[1, 1].set_title('FFT')
    axes[1, 1].set_xlabel('Frequency [Hz]')
    axes[1, 1].set_ylabel('Power')
    axes[1, 1].set_xlim(0, target_freq_1 * 2)

    plt.tight_layout()
    plt.show()

    plt.close()
    plt.clf()


def windowing():
    fig = plt.figure()
    axes = fig.subplots(3, 1)

    target_freq_0 = 8.0
    target_freq_1 = 8.5

    sampling_rate = 1000.0
    duration = 1.0

    dt = 1.0 / sampling_rate

    n_samples = int(duration / dt) + 1

    print(f'duration: {duration} [sec], dt: {dt} [sec], sampling_rate: {sampling_rate} [Hz]')

    spectrum_size = n_samples * 4
    padding = spectrum_size - n_samples
    print(f'spectrum_size: {spectrum_size}, padding: {padding}')

    window = torch.kaiser_window(n_samples, periodic=False, beta=8.0)
    print(f'window.shape: {window.shape}, window.min(): {window.min()}, window.max(): {window.max()}')
    window *= window.square().sum().rsqrt()
    print(f'window.shape: {window.shape}, window.min(): {window.min()}, window.max(): {window.max()}')
    windows = window.ger(window)
    print(f'window.shape: {window.shape}, window.min(): {window.min()}, window.max(): {window.max()}')

    x = torch.linspace(0.0, duration, n_samples)
    y = torch.sin(2.0 * np.pi * target_freq_0 * x)

    # ------------------------------------------------------------------------
    # Raw signal
    axes[0].plot(x, y)
    axes[0].set_title(r'$y = \sin(2\pi \cdot {} \cdot x)$'.format(target_freq_0))
    axes[0].set_xlabel('t [sec]')
    axes[0].set_ylabel('y')

    # ------------------------------------------------------------------------
    # Window
    axes[1].plot(x, window)
    axes[1].set_title('Window')
    axes[1].set_xlabel('t [sec]')
    axes[1].set_ylabel('Window')

    # ------------------------------------------------------------------------
    # Windowed signal
    y_windowed = y * window
    axes[2].plot(x, y_windowed)
    axes[2].set_title('Windowed signal')
    axes[2].set_xlabel('t [sec]')
    axes[2].set_ylabel('y')

    fig.tight_layout()
    plt.show()


def fft_1d_with_window():
    fig = plt.figure()
    axes = fig.subplots(2, 3)

    target_freq_0 = 8.0
    target_freq_1 = 8.5

    sampling_rate = 1000.0
    duration = 1.0

    dt = 1.0 / sampling_rate

    n_samples = int(duration / dt) + 1

    print(f'duration: {duration} [sec], dt: {dt} [sec], sampling_rate: {sampling_rate} [Hz]')

    spectrum_size = n_samples * 4
    padding = spectrum_size - n_samples
    print(f'spectrum_size: {spectrum_size}, padding: {padding}')

    window = torch.kaiser_window(n_samples, periodic=False, beta=8.0)
    window *= window.square().sum().rsqrt()
    windows = window.ger(window)

    # ------------------------------------------------------------------------
    # Raw signal
    x = torch.linspace(0.0, duration, n_samples)
    y = torch.sin(2.0 * np.pi * target_freq_0 * x)
    axes[0, 0].plot(x, y)
    axes[0, 0].set_title(r'$y = \sin(2\pi \cdot {} \cdot x)$'.format(target_freq_0))
    axes[0, 0].set_xlabel('t [sec]')
    axes[0, 0].set_ylabel('y')

    # Windowing
    y_windowed = torch.nn.functional.pad(y * window, (padding, padding))
    print(f'y_windowed.shape: {y_windowed.shape}, y_windowed.min(): {y_windowed.min()}, y_windowed.max(): {y_windowed.max()}')
    axes[0, 1].plot(y_windowed)
    axes[0, 1].set_title('Windowed signal')
    axes[0, 1].set_ylabel('y')

    # FFT
    freq = torch.fft.fftfreq(y_windowed.shape[-1], d=dt)
    fourier_coef = torch.fft.fft(y_windowed)

    power = torch.abs(fourier_coef) ** 2
    axes[0, 2].plot(freq[1:y_windowed.shape[-1] // 2], power[1:y_windowed.shape[-1] // 2])
    axes[0, 2].set_title('FFT')
    axes[0, 2].set_xlabel('Frequency [Hz]')
    axes[0, 2].set_ylabel('Power')
    axes[0, 2].set_xlim(0, target_freq_0 * 2)

    # ------------------------------------------------------------------------
    # Raw signal
    x = torch.linspace(0.0, duration, n_samples)
    y = torch.sin(2.0 * np.pi * target_freq_1 * x)
    axes[1, 0].plot(x, y)
    axes[1, 0].set_title(r'$y = \sin(2\pi \cdot {} \cdot x)$'.format(target_freq_1))
    axes[1, 0].set_xlabel('t [sec]')
    axes[1, 0].set_ylabel('y')

    # Windowing
    y_windowed = torch.nn.functional.pad(y * window, (padding, padding))
    print(f'y_windowed.shape: {y_windowed.shape}, y_windowed.min(): {y_windowed.min()}, y_windowed.max(): {y_windowed.max()}')
    axes[1, 1].plot(y_windowed)
    axes[1, 1].set_title('Windowed signal')
    axes[1, 1].set_ylabel('y')

    # FFT
    freq = torch.fft.fftfreq(y_windowed.shape[-1], d=dt)
    fourier_coef = torch.fft.fft(y_windowed)

    power = torch.abs(fourier_coef) ** 2
    axes[1, 2].plot(freq[1:y_windowed.shape[-1] // 2], power[1:y_windowed.shape[-1] // 2])
    axes[1, 2].set_title('FFT')
    axes[1, 2].set_xlabel('Frequency [Hz]')
    axes[1, 2].set_ylabel('Power')
    axes[1, 2].set_xlim(0, target_freq_0 * 2)

    fig.tight_layout()
    plt.show()


def main():
    # fft_1d()
    # windowing()
    fft_1d_with_window()


if __name__ == '__main__':
    main()
