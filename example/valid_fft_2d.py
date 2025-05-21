import matplotlib.pyplot as plt
import PIL.Image
import torch
import torchvision.transforms.functional as F

import research_utilities as _rutils


def save_img(tensor, filename):
    min_val = tensor.min()
    max_val = tensor.max()

    F.to_pil_image((tensor.detach().clone().cpu().squeeze() - min_val) / (max_val - min_val)).save(filename)


def main():
    # img = F.to_tensor(PIL.Image.open(_rutils.get_demo_image())).unsqueeze(0)
    img = F.to_tensor(PIL.Image.open(_rutils.get_checkerboard_image()))
    short_size = min(img.shape[-2:])
    img = F.center_crop(img, [short_size, short_size])
    img = 2.0 * img - 1.0
    print(f'img.shape: {img.shape}, (min, max): ({img.min()}, {img.max()})')

    # Compute power spectral density (PSD)
    psd = _rutils.compute_psd(img, beta=8.0, padding_factor=4)
    psd = psd.mean(dim=(0, 1))
    psd = 20.0 * torch.log10(psd + 1e-10)  # Convert to dB
    print(f'psd.shape: {psd.shape}, (min, max): ({psd.min()}, {psd.max()})')
    save_img(psd, 'fft.png')

    # Compute radial PSD
    radial_psd = _rutils.compute_radial_psd(img)
    radial_psd = radial_psd.mean(dim=(0, 1, 2))
    # radial_psd = 20.0 * torch.log10(radial_psd + 1e-10)  # Convert to dB
    print(f'radial_psd.shape: {radial_psd.shape}, (min, max): ({radial_psd.min()}, {radial_psd.max()})')

    freq = _rutils.radial_freq(img.shape[-1], radial_psd.shape[-1])
    plt.plot(freq[1:], radial_psd[1:])
    plt.title('Radial PSD')
    plt.xlabel('Frequency [1/pixel]')
    plt.ylabel('Power [dB]')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.savefig('radial_psd.png')


if __name__ == '__main__':
    main()
