"""Utility functions for color maps
"""

import pathlib
import random
import typing

import cv2
import matplotlib as mpl
import matplotlib.colors as mplib_colors
import numpy as np
import torch
import typing_extensions

# ------------------------------------------------------------------------------------------------------------
# Lookup table for color maps

_cmap_luts: dict[str, np.ndarray] = {}

_cmap_lut_file = pathlib.Path(__file__).parent / 'cmap' / 'cmap_luts.npz'
if _cmap_lut_file.exists():
    with np.load(_cmap_lut_file) as data:
        for key, value in data.items():
            _cmap_luts[key] = value

# ------------------------------------------------------------------------------------------------------------
# Interpolation function
#
# Imported from: https://github.com/pytorch/pytorch/issues/50334#issuecomment-1247611276


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int = -1, extrapolate: str = 'constant') -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == 'constant':
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else:  # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values.movedim(-1, dim)

# ------------------------------------------------------------------------------------------------------------
# Converting functions


ArrayLike = typing.TypeVar('ArrayLike', np.ndarray, torch.Tensor)


def apply_color_map(input_img: ArrayLike, color_map_type: str = 'viridis') -> ArrayLike:
    assert input_img.ndim in (2, 3), "Input image must be a 2D (HW) or 3D (BHW) array."
    assert color_map_type in _cmap_luts, f"Color map '{color_map_type}' is not supported. You can choose from: {list(_cmap_luts.keys())}"

    is_np = isinstance(input_img, np.ndarray)
    img = torch.tensor(input_img) if is_np else input_img

    has_batch_dim = (img.ndim == 3)
    if not has_batch_dim:
        img = img.unsqueeze(0)  # Add batch dimension
    assert img.ndim == 3

    n_data, height, width = img.shape
    orig_dtype = img.dtype

    img = img.float()  # Convert to float for processing
    img = img.flatten(1)  # [n_data, height * width]

    orig_min = img.min(dim=1, keepdim=True).values
    orig_max = img.max(dim=1, keepdim=True).values

    # Normalize the image to [0, 1]
    img = (img - orig_min) / (orig_max - orig_min + 1e-8)  # Avoid division by zero

    # Apply the colormap (linear interpolation)
    lut = _cmap_luts[color_map_type]
    lut = torch.tensor(lut, dtype=img.dtype, device=img.device)
    n_points = lut.shape[0]

    xp = torch.linspace(0.0, 1.0, n_points, dtype=img.dtype, device=img.device)
    fp_r = lut[:, 0]  # Red channel
    fp_g = lut[:, 1]  # Green channel
    fp_b = lut[:, 2]  # Blue channel

    flattened_img = img.reshape(-1)
    colorred = torch.stack(
        (
            interp(flattened_img, xp, fp_r, dim=-1, extrapolate='constant').reshape(n_data, -1),
            interp(flattened_img, xp, fp_g, dim=-1, extrapolate='constant').reshape(n_data, -1),
            interp(flattened_img, xp, fp_b, dim=-1, extrapolate='constant').reshape(n_data, -1)
        ),
        dim=-1
    )  # [n_data, height * width, 3]

    # Convert back to original range
    colorred = colorred * (orig_max.unsqueeze(-1) - orig_min.unsqueeze(-1)) + orig_min.unsqueeze(-1)

    # Convert back to original dtype
    colorred = colorred.to(orig_dtype)

    # Reshape back to original shape
    colorred = colorred.reshape(n_data, height, width, 3)  # [n_data, height, width, 3]

    if not has_batch_dim:
        colorred = colorred.squeeze(0)

    if is_np:
        colorred = colorred.detach().cpu().numpy()

    return colorred


@typing_extensions.deprecated('This function is deprecated and was renamed from "apply_color_map".')
def apply_color_map_v1(img: np.ndarray, color_map_type: str = 'viridis') -> np.ndarray:
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


# ------------------------------------------------------------------------------------------------------------


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
