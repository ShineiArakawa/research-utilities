import dataclasses
import functools
import logging
import os
import pathlib
import platform
import shutil
import sys
import typing

import torch
import torch.utils.cpp_extension as _cpp_extension

import research_utilities.common as _common


@dataclasses.dataclass
class ExtensionSpec:
    name: str
    sources: list[str]
    src_root_dir: str
    verbose: bool
    debug: bool


class ExtensionLoader:
    def __init__(self, src_dir: str = 'csrc'):
        self._logger = _common.get_logger()

        self.extensions: typing.Dict = {}
        self.extension_spec: typing.Dict[str, ExtensionSpec] = {}

        self.src_root_dir = (pathlib.Path(src_dir) if os.path.isabs(src_dir) else (pathlib.Path(__file__).parent / src_dir)).resolve()

        # Set the CUDA architecture
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
            compute_capability = torch.cuda.get_device_capability(device)
            str_cuda_arch = f'{compute_capability[0]}.{compute_capability[1]}'
            self._logger.info(f'Using CUDA architecture: {str_cuda_arch}')
            os.environ['TORCH_CUDA_ARCH_LIST'] = str_cuda_arch

    def _check_command(self, command: str) -> bool:
        """
        Check if a command is available in the system PATH.
        """
        if shutil.which(command) is None:
            self._logger.warning(f'Command \'{command}\' not found in PATH.')
            return False
        return True

    def load(
        self,
        name: str,
        sources: list[str],
        src_root_dir: str | None = None,
        verbose: bool = False,
        debug: bool = False
    ):
        extension_spec = ExtensionSpec(
            name=name,
            sources=sources,
            src_root_dir=src_root_dir,
            verbose=verbose,
            debug=debug
        )

        # Check if the extension is already loaded
        if name in self.extensions:
            return self.extensions[name]

        self._logger.info(f'Building \'{name}\' ... ')

        # Check if the sources are relative to the src_root_dir
        if src_root_dir is None:
            src_root_dir = self.src_root_dir

        # Convert the sources to absolute paths
        sources = [str(src_root_dir / source) for source in sources]

        # Check if the sources contain CUDA files
        with_cuda = any(
            [source.endswith('.cu') for source in sources]
        ) or any(
            [source.endswith('.cuh') for source in sources]
        )

        # Set the extra flags
        is_windows = platform.system() == 'Windows'
        if is_windows:
            extra_cflags = ['/O2']
            extra_cuda_cflags = ['/O2']
        else:
            extra_cflags = ['-O3']
            extra_cuda_cflags = ['-O3']

        if debug:
            # 'DEBUG_MODE' is defined in the C++ and cuda code
            if is_windows:
                extra_cflags += ['/DDEBUG_MODE']
                extra_cuda_cflags += ['/DDEBUG_MODE']
            else:
                extra_cflags += ['-DDEBUG_MODE']
                extra_cuda_cflags += ['-DDEBUG_MODE']

        # build directory
        pycache_dir = pathlib.Path(sys.pycache_prefix) if sys.pycache_prefix else pathlib.Path(__file__).parent / '__pycache__'
        build_dir = pycache_dir / 'torch_extensions' / name
        build_dir.mkdir(parents=True, exist_ok=True)

        # Add include directories
        extra_include_paths = [str(src_root_dir)]

        module = _cpp_extension.load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            build_directory=str(build_dir),
            verbose=verbose,
            with_cuda=with_cuda,

        )

        self.extensions[name] = module
        self.extension_spec[name] = extension_spec

        return module


@functools.lru_cache(maxsize=None)
def get_extension_loader(*args, **kwargs):
    return ExtensionLoader(*args, **kwargs)
