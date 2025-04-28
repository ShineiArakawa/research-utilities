import cv2
import torch
from matplotlib import pyplot as plt

import research_utilities.common as _common
import research_utilities.demo_imgs as _demo_imgs
import research_utilities.signal as _signal


def main(with_cuda: bool = True) -> None:
    logger = _common.get_logger()

    # Download an demo image
    file_path = _demo_imgs.get_demo_image()

    # Load the image
    img_np = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    img = torch.from_numpy(img_np).to(torch.float32)
    img = img / 255.0
    logger.info(f'img.shape: {img.shape}, img.dtype: {img.dtype}')

    if with_cuda:
        # Move the image to GPU
        img = img.cuda()

    radial_prof = _signal.calc_radial_psd_profile(img, n_divs=360, n_points=512)
    logger.info(f'radial_prof.shape: {radial_prof.shape}, radial_prof.dtype: {radial_prof.dtype}')

    radial_prof = radial_prof.squeeze(0).squeeze(0).cpu().numpy()
    radial_prof = radial_prof.mean(axis=0)  # mean over the radial dimension

    # Plot the radial PSD profile
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    ax.plot(radial_prof, label='Radial PSD Profile')
    ax.set_xlabel('Frequency (cycles/pixel)')
    ax.set_ylabel('Power Spectral Density (PSD)')
    ax.set_title('Radial Power Spectral Density Profile')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()

    fname = 'radial_psd_profile_{}.png'.format('cuda' if with_cuda else 'cpu')
    fig.savefig(fname)


if __name__ == '__main__':
    main(with_cuda=True)
    main(with_cuda=False)
