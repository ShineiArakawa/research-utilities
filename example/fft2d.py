import cv2
import numpy as np
from matplotlib import pyplot as plt

import research_utilities as rutils


def main():
    logger = rutils.get_logger()

    # ------------------------------------------------------------
    # Create a test image

    img_size = 128

    nyquist_freq = 0.5  # [cycles/pixel]

    # Create a sinusoidal image
    target_freq = 20.3

    target_freq = target_freq / img_size   # [cycles/pixel]
    # target_freq = target_freq / (img_size - 1)  # [cycles/pixel]

    # target_freq = nyquist_freq / 2.0  # [cycles/pixel]

    x = np.arange(img_size, dtype=np.float64)
    y = np.sin(2.0 * np.pi * target_freq * x)

    img = np.tile(y, (img_size, 1))

    cv2.imwrite('sinusoidal_image.png', (img * 127.5 + 127.5).astype(np.uint8))

    img_tile = np.hstack((img, img))
    cv2.imwrite('sinusoidal_image_tiled.png', (img_tile * 127.5 + 127.5).astype(np.uint8))

    # ------------------------------------------------------------
    # Create a 2D FFT of the image

    psd = rutils.compute_psd(img, is_db_scale=True, beta=8.0, interpolation=4)
    psd = psd.squeeze(0).squeeze(0)

    freqs = np.fft.fftfreq(img_size)
    freqs = np.fft.fftshift(freqs)

    plt.imshow(psd, cmap='viridis', extent=[-img_size // 2, img_size // 2, -img_size // 2, img_size // 2])
    # plt.imshow(psd)
    # plt.xlabel('Frequency [cycles/pixel]')
    # plt.ylabel('Frequency [cycles/pixel]')
    # plt.title('2D FFT Power Spectrum')
    # plt.colorbar()
    # plt.savefig('fft2d_spectrum.png')
    plt.show()
    plt.clf()

    # ------------------------------------------------------------
    # Compute radial average of the power spectrum

    rad_psd = rutils.compute_radial_psd(img, n_divs=720, n_points=img.shape[-1] * 2)
    rad_psd = rad_psd.squeeze(0).squeeze(0).mean(axis=0)

    freqs = rutils.radial_freq(img_size, len(rad_psd))

    plt.plot(freqs, 20.0 * rad_psd)  # Ignore the DC
    plt.yscale('log')
    plt.xlabel('Frequency [cycles/pixel]')
    plt.ylabel('Power Spectrum Density [db]')
    plt.title('Radial Average of 2D FFT Power Spectrum')
    plt.grid(axis='both', which='major')
    plt.grid(axis='both', which='minor', alpha=0.2)
    plt.savefig('fft2d_radial_spectrum.png', dpi=500)


if __name__ == '__main__':
    main()
