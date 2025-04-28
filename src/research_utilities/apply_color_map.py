import random

import cv2
import matplotlib as mpl
import matplotlib.colors as mplib_colors
import numpy as np

import research_utilities.common as _common


def apply_color_map(img: np.ndarray, color_map_type: str) -> np.ndarray:
    """Apply color map to an image.

    The input image must be a single channel image.

    Parameters
    ----------
    img : np.ndarray
        The grayscale image to be colorized. The input image must be a two-dimensional array.
    color_map_type : str
        The color map to be applied to the image.

    Returns
    -------
    np.ndarray
        The colorized image. The output image is in BGR format.
    """

    assert img.ndim == 2

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
        colorred = color_map(img_scaled)  # colorred is in HWC format

        # Convert the image to the original dtype and scale
        colorred = (colorred * (orig_max - orig_min) + orig_min).astype(orig_dtype)

        colorred = cv2.cvtColor(colorred, cv2.COLOR_RGB2BGR)
    else:
        # NOTE: use cv2 colormap
        # see: https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html

        color_map = getattr(cv2, color_map_type, default=cv2.COLORMAP_HOT)

        colorred = cv2.applyColorMap(img, color_map)

    return colorred


class CyclicColorIterator(object):
    """Cyclic color iterator
    """

    def __init__(self, is_massive: bool) -> None:
        """Initialize the iterator

        Parameters
        ----------
        is_massive : bool
            Use larger color set
        """

        self._base_colors = mplib_colors.XKCD_COLORS if is_massive else mplib_colors.CSS4_COLORS
        self._keys = list(self._base_colors.keys())
        self._counter = 0

        random.shuffle(self._keys)

    def __iter__(self):
        return self

    def __next__(self):
        self._counter += 1

        if self._counter >= len(self._keys):
            self._counter = 0
            pass

        return self._base_colors[self._keys[self._counter]]
