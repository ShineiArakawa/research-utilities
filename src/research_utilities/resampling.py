import enum
import functools

import numpy as np
import torch

import research_utilities.common as _common
import research_utilities.torch_util as _torch_util


@functools.lru_cache(maxsize=None)
def _get_cpp_module():
    ext_loader = _torch_util.get_extension_loader()

    module = ext_loader.load(
        name='resampling',
        sources=[
            'resampling.cpp',
            'resampling.cu',
        ],
        debug=_common.GlobalSettings.DEBUG_MODE
    )

    return module


class InterpMethod(str, enum.Enum):
    NEAREST = 'nearest'
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    LANCZOS4 = 'lanczos4'

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def get_enum(self):
        _module = _get_cpp_module()

        if self == InterpMethod.NEAREST:
            return _module.InterpMethod.NEAREST
        elif self == InterpMethod.BILINEAR:
            return _module.InterpMethod.BILINEAR
        elif self == InterpMethod.BICUBIC:
            return _module.InterpMethod.BICUBIC
        elif self == InterpMethod.LANCZOS4:
            return _module.InterpMethod.LANCZOS4
        else:
            raise ValueError(f'Unknown interpolation method: {self}')


def resample(
    img: torch.Tensor,
    scale_factor: float,
    interp_method: InterpMethod = InterpMethod.BILINEAR
) -> torch.Tensor:
    """Resample an image using the specified interpolation method.
    The input image must be on the GPU.

    Parameters
    ----------
    img : torch.Tensor
        Input image to be resampled. The input image must be on the GPU.
        The input image can have 2, 3, or 4 dimensions.
        - 2D image: [height, width]
        - 3D image: [height, width, channels]
        - 4D image: [batch, height, width, channels]
    scale_factor : float
        Scale factor for resampling. The output image will have the shape
        [batch, out_height, out_width, channels], where
        out_height = height * scale_factor
        out_width = width * scale_factor
        The output image will be in the same format as the input image.
    interp_method : InterpMethod, optional
        Interpolation method to be used for resampling. The default is
        InterpMethod.BILINEAR. The available methods are:
        - InterpMethod.NEAREST
        - InterpMethod.BILINEAR
        - InterpMethod.BICUBIC
        - InterpMethod.LANCZOS4
        The default is InterpMethod.BILINEAR.

    Returns
    -------
    torch.Tensor
        Resampled image. The output image will have the same format as the input image.
        The output image will be on the GPU.

    Raises
    ------
    ValueError
        If the input image is not on the GPU or if the input image has
        an invalid number of dimensions.
    """

    assert img.is_cuda, 'The input image must be on the GPU.'

    # Check the input
    if img.ndim == 2:
        # Add the batch and channel dimensions
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        # Add the batch dimension
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError(f'Input image must have 2, 3, or 4 dimensions, but got {img.ndim}.')

    # Result shape
    out_height = int(img.shape[2] * scale_factor)
    out_width = int(img.shape[3] * scale_factor)

    # Mesh grid
    x = np.linspace(0.0, 1.0, out_width, dtype=np.float32)   # width
    y = np.linspace(0.0, 1.0, out_height, dtype=np.float32)  # height
    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')
    grid_base = np.stack([grid_x, grid_y], axis=-1)  # Grid is the order of (width, height) because we assume the uv coordinate system

    grid = torch.from_numpy(grid_base).to(img.device)
    grid = grid.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)  # [batch, out_height, out_width, 2]

    assert grid.shape[1] == out_height
    assert grid.shape[2] == out_width

    # Load the extension
    _module = _get_cpp_module()

    # Convert to channels last
    img = img.permute(0, 2, 3, 1).contiguous()

    # Interpolate
    out = _module.resample_image_cuda(img, grid, interp_method.get_enum())

    # Convert back to channels first
    out = out.permute(0, 3, 1, 2).contiguous()

    return out
