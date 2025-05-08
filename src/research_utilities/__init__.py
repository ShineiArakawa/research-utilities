"""
Research Utilities
==================

A collection of utility functions and classes for research workflows,
including plotting, signal processing, multiprocessing, and image utilities.

Modules
-------

- cmap: Color mapping for visualization
- common: Logging utilities
- demo_imgs: Provides demo images
- multi_processing: Multiprocessing support
- plotting: Plot customization
- resampling: Resampling and interpolation
- signal: Signal processing and FFT tools

Contact
-------

- Author: Shinei Arakawa
- Email: sarakawalab@gmail.com

"""

# ----------------------------------------------------------------------------
# Check Python version

import sys

if sys.version_info < (3, 10):
    raise ImportError("Python 3.10 or higher is required.")

# ----------------------------------------------------------------------------
# Check the version of this package

import importlib.metadata

try:
    __version__ = importlib.metadata.version("research_utilities")
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

# ----------------------------------------------------------------------------
# Import modules

from .cmap import apply_color_map
from .common import get_logger
from .demo_imgs import get_checkerboard_image, get_demo_image
from .multi_processing import launch_multi_process
from .plotting import add_title
from .resampling import InterpMethod, resample
from .signal import (calc_radial_psd_profile, compute_radial_psd, fft_1d,
                     fft_2d, plot_signal)

__all__ = [
    "apply_color_map",
    "get_logger",
    "get_checkerboard_image",
    "get_demo_image",
    "launch_multi_process",
    "add_title",
    "InterpMethod",
    "resample",
    "calc_radial_psd_profile",
    "compute_radial_psd",
    "fft_1d",
    "fft_2d",
    "plot_signal",
]
