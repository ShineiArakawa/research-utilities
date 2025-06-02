import logging

import cv2


def test_apply_color_map(test_image: str, logger: logging.Logger):
    from src.research_utilities.cmap import apply_color_map

    logger.info(f'test_image: {test_image}')

    # Load the test image
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Failed to load the test image."

    # Apply the color map
    color_map_type = 'viridis'
    colorized_img = apply_color_map(img, color_map_type)

    logger.debug(f'colorized_img.shape: {colorized_img.shape}')
    logger.debug(f'colorized_img.dtype: {colorized_img.dtype}')
    logger.debug(f'colorized_img.min(): {colorized_img.min()}, colorized_img.max(): {colorized_img.max()}')

    # Optionally, save or display the colorized image for verification
    # cv2.imwrite('colorized_image.png', colorized_img.numpy())
