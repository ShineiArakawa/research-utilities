"""This module provides a class to load C++/CUDA extensions for PyTorch.
"""

import dataclasses
import datetime
import functools
import glob
import hashlib
import json
import os
import pathlib
import platform
import shutil
import sys
import sysconfig
import typing

import torch
import torch.utils.cpp_extension as _cpp_extension

import research_utilities.common as _common


@dataclasses.dataclass(frozen=True)
class ExtensionSpec:
    """Extension specification to uniquely identify a loaded extension.
    """

    # autopep8: off
    name         : str
    sources      : tuple[str, ...]
    src_root_dir : str
    with_omp     : bool
    verbose      : bool
    debug        : bool
    # autopep8: on

    def deterministic_hash(self) -> str:
        """Generate a deterministic hash for the extension specification.
        This hash is used to uniquely identify the extension.

        The function 'hash', which is provided by Python standard library, yields different values for the same object in different sessions.
        So, we use 'hashlib' which can generate a unique hash for the same object across different sessions.
        """

        spec_json = json.dumps(dataclasses.asdict(self), sort_keys=True)
        byte_codes = spec_json.encode('utf-8')

        return hashlib.sha256(byte_codes).hexdigest()


@dataclasses.dataclass(frozen=True)
class Extension:
    # autopep8: off
    spec       : ExtensionSpec
    module     : typing.Any
    build_dir  : pathlib.Path
    timestamp  : str
    # autopep8: on


class ExtensionLoader:
    def __init__(self, src_dir: str = 'csrc'):
        # self._logger = _common.get_logger()
        import logging
        self._logger = logging.getLogger(__name__)

        self.extensions: dict[str, Extension] = {}

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
    ) -> typing.Any:
        extension_spec = ExtensionSpec(
            name=name,
            sources=tuple(sources),
            src_root_dir=src_root_dir,
            with_omp=with_omp,
            verbose=verbose,
            debug=debug
        )

        extension_hash = extension_spec.deterministic_hash()

        # Check if the extension is already loaded
        if extension_hash in self.extensions:
            return self.extensions[extension_hash].module

        # --------------------------------------------------------------------------
        # Build the extension
        self._logger.info(f'Building \'{name}\' ... ')
        self._logger.debug(f'Extension spec: {extension_spec}')

        platform_name = platform.system()

        if platform_name == 'Windows':
            # Find cl.exe and add it to the PATH
            machine_arch = sysconfig.get_platform().split('-')[-1].lower()
            assert machine_arch in ['amd64', 'arm64'], f'Unsupported machine architecture: {machine_arch}'

            msvc_arch_name = 'x64' if machine_arch == 'amd64' else 'arm64'
            msvc_bin_dir_pattern = f'C:\\Program Files*\\Microsoft Visual Studio\\*\\*\\VC\\Tools\\MSVC\\*\\bin\\Host{msvc_arch_name}\\{msvc_arch_name}'

            matched_dirs = glob.glob(msvc_bin_dir_pattern)

            if len(matched_dirs) == 0:
                msvc_bin_dir = None
                self._logger.warning(f'No MSVC bin directory found for {msvc_arch_name}.')
            elif len(matched_dirs) == 1:
                msvc_bin_dir = matched_dirs[0]
                self._logger.debug(f'Found MSVC bin directory: {msvc_bin_dir}')
            elif len(matched_dirs) > 1:
                msvc_bin_dir = matched_dirs[-1]
                self._logger.warning(f'Multiple MSVC bin directories found for {msvc_arch_name}: {matched_dirs}')
                self._logger.warning(f'Using the last one: {msvc_bin_dir}')

            if msvc_bin_dir is not None:
                # Add the MSVC bin directory to the PATH
                os.environ['PATH'] = f'{msvc_bin_dir};{os.environ["PATH"]}'

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
        build_dir = pycache_dir / 'torch_extensions' / extension_hash
        build_dir.mkdir(parents=True, exist_ok=True)

        self._logger.debug(f'Build directory: {build_dir}')

        # Add include directories
        extra_include_paths = [str(src_root_dir)]

        # -----------------------------------------------------------------------------
        # Build the extension
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

        # -----------------------------------------------------------------------------
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

        extension = Extension(
            spec=extension_spec,
            module=module,
            build_dir=build_dir,
            timestamp=timestamp
        )

        with open(build_dir / 'extension_spec.json', 'w') as file:
            to_save = {
                'build_dir': str(build_dir),
                'timestamp': timestamp,
                'spec': dataclasses.asdict(extension.spec),
            }
            json.dump(to_save, file, indent=4)

        # -----------------------------------------------------------------------------
        # Cache the extension
        self.extensions[extension_hash] = extension

        return module


@functools.lru_cache(maxsize=None)
def get_extension_loader(*args, **kwargs):
    return ExtensionLoader(*args, **kwargs)
