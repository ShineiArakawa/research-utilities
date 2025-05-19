import cv2
import numpy as np

import research_utilities.common as _common
import research_utilities.demo_imgs as _demo_imgs
import research_utilities.signal as _signal


def main():
    logger = _common.get_logger()

    # Download an demo image
    file_path = _demo_imgs.get_demo_image()

    # Load the image
    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    logger.info(f'img.shape: {img.shape}, img.dtype: {img.dtype}')
    logger.info(f'img.min(): {img.min()}, img.max(): {img.max()}')

    # Apply FFT
    _signal.fft_2d(img, 'fft.png')


if __name__ == '__main__':
    main()
