"""This module provides a class to load C++/CUDA extensions for PyTorch.

メモ： gcc/g++ + CUDA + OpenMPでのみ動作することを確認した
"""

import dataclasses
import functools
import os
import pathlib
import platform
import shutil
import sys
import typing
import uuid

import torch
import torch.utils.cpp_extension as _cpp_extension

import research_utilities.common as _common


@dataclasses.dataclass
class ExtensionSpec:
    # autopep8: off
    name         : str
    sources      : list[str]
    src_root_dir : str
    with_omp     : bool
    verbose      : bool
    debug        : bool
    # autopep8: on


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
        with_omp: bool = False,
        verbose: bool = False,
        debug: bool = False
    ):
        extension_spec = ExtensionSpec(
            name=name,
            sources=sources,
            src_root_dir=src_root_dir,
            with_omp=with_omp,
            verbose=verbose,
            debug=debug
        )

        # Check if the extension is already loaded
        matched_ext_id = None
        for ext_id in self.extensions.keys():
            spec = self.extension_spec[ext_id]
            if extension_spec == spec:
                matched_ext_id = ext_id
                break

        if matched_ext_id is not None:
            return self.extensions[matched_ext_id]

        # Build the extension
        self._logger.info(f'Building \'{name}\' ... ')
        self._logger.debug(f'Extension spec: {extension_spec}')

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

        extra_cflags = []
        extra_cuda_cflags = []
        extra_ldflags = []

        if with_cuda:
            # Check if nvcc is available
            if self._check_command('nvcc'):
                extra_cflags += ['-DWITH_CUDA']
                extra_cuda_cflags += ['-DWITH_CUDA']
            else:
                self._logger.warning('nvcc command not found. CUDA toolkit is properly installed?')
                self._logger.warning('Falling back to CPU only build, excluding CUDA files.')

                sources = [source for source in sources if (not source.endswith('.cu') and (not source.endswith('.cuh')))]

                with_cuda = False

        # Set the extra flags
        platform_name = platform.system()
        if platform_name == 'Windows':
            extra_cflags += ['/O2']
            extra_cuda_cflags += ['/O2']
        else:
            extra_cflags += ['-O3']
            extra_cuda_cflags += ['-O3']

        # Add OpenMP flags
        if with_omp:
            if platform_name == 'Windows':
                extra_cflags += ['/openmp']
            elif platform_name == 'Darwin':
                # macOS: Assume Homebrew clang (brew install libomp)
                extra_cflags += ['-Xpreprocessor', '-fopenmp']
                extra_ldflags += ['-lomp']
            else:
                # Linux
                extra_cflags += ['-fopenmp']

            extra_cflags += ['-DWITH_OPENMP']
            extra_cuda_cflags += ['-DWITH_OPENMP']
        else:
            extra_ldflags += []

        if debug:
            # 'DEBUG_MODE' is defined in the C++ and cuda code
            if platform_name == 'Windows':
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
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=str(build_dir),
            verbose=verbose,
            with_cuda=with_cuda,
        )

        ext_id = str(uuid.uuid4())
        self.extensions[ext_id] = module
        self.extension_spec[ext_id] = extension_spec

        return module


@functools.lru_cache(maxsize=None)
def get_extension_loader(*args, **kwargs):
    return ExtensionLoader(*args, **kwargs)
