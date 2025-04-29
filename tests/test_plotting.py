import logging

import cv2

import research_utilities.plotting as _plotting


def test_add_title(test_image: str, logger: logging.Logger) -> None:
    img = cv2.imread(test_image, cv2.IMREAD_COLOR_BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    logger.debug(f'img.shape: {img.shape}')

    out_img = _plotting.add_title(
        img,
        'Hello World',
        (0, 0, 0, 255),
        (255, 255, 255, 0),
    )

    cv2.imwrite('out_img.png', out_img)
