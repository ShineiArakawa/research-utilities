import cv2
import matplotlib as mpl
import numpy as np


def apply_color_map(img: np.ndarray, color_map_type: str) -> np.ndarray:
    """Apply color map to an image.

    The input image must be a single channel image.

    Parameters
    ----------
    img : np.ndarray
        The grayscale image to be colorized.
    color_map_type : str
        The color map to be applied to the image.

    Returns
    -------
    np.ndarray
        The colorized image. The output image is in BGR format.
    """

    colorred: np.ndarray = None

    orig_dtype = img.dtype
    orig_min: float = img.min()
    orig_max: float = img.max()

    if color_map_type in mpl.colormaps:
        # NOTE: use matplotlib colormap
        # see: https://matplotlib.org/stable/tutorials/colors/colormaps.html

        color_map = mpl.colormaps[color_map_type]

        # Normalize the image to [0, 1]
        img_scaled = (img.astype(np.float32) - orig_min) / (orig_max - orig_min)

        # Apply the colormap
        colorred = color_map(img_scaled)

        # Convert the image to the original dtype and scale
        colorred = (colorred * (orig_max - orig_min) + orig_min).astype(orig_dtype)

        # squeeze
        if colorred.shape[-2] == 1:
            colorred = colorred.squeeze(-2)

        colorred = cv2.cvtColor(colorred, cv2.COLOR_RGB2BGR)
    else:
        # NOTE: use cv2 colormap
        # see: https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html

        color_map = getattr(cv2, color_map_type, default=cv2.COLORMAP_HOT)

        colorred = cv2.applyColorMap(img, color_map)

    return colorred
