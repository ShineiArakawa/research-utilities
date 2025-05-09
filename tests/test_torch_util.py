import enum
import functools
import logging
import random
import typing

import research_utilities.torch_util as _torch_util


def test_exntension_loading(logger: logging.Logger) -> None:
    class ModuleType(str, enum.Enum):
        signal = 'signal'
        resampling = 'resampling'

        def __str__(self):
            return self.value

    def load_signal_module(with_cuda: bool, with_omp: bool, debug: bool) -> typing.Any:
        ext_loader = _torch_util.get_extension_loader()

        name = 'signal'
        sources = [
            'signal.cpp',
        ]

        if with_cuda:
            sources += [
                'signal.cu',
                'resampling.cu',
            ]
            name += '_cuda'

        module = ext_loader.load(
            name=name,
            sources=sources,
            debug=debug,
            with_omp=with_omp,
        )

        return module

    def load_resampling_module(with_omp: bool, debug: bool) -> typing.Any:
        ext_loader = _torch_util.get_extension_loader()

        module = ext_loader.load(
            name='resampling',
            sources=[
                'resampling.cpp',
                'resampling.cu',
            ],
            with_omp=with_omp,
            debug=debug
        )

        return module

    recipe = [
        functools.partial(load_signal_module, with_cuda=False, with_omp=False, debug=False),
        functools.partial(load_signal_module, with_cuda=True, with_omp=False, debug=False),
        functools.partial(load_signal_module, with_cuda=False, with_omp=True, debug=False),
        functools.partial(load_signal_module, with_cuda=True, with_omp=True, debug=False),
        functools.partial(load_signal_module, with_cuda=False, with_omp=False, debug=True),
        functools.partial(load_signal_module, with_cuda=True, with_omp=False, debug=True),
        functools.partial(load_signal_module, with_cuda=False, with_omp=True, debug=True),
        functools.partial(load_signal_module, with_cuda=True, with_omp=True, debug=True),
        functools.partial(load_resampling_module, with_omp=False, debug=False),
        functools.partial(load_resampling_module, with_omp=True, debug=False),
        functools.partial(load_resampling_module, with_omp=False, debug=True),
        functools.partial(load_resampling_module, with_omp=True, debug=True),
    ]

    NUM_TRIES = 100

    trial_id = random.choices(range(len(recipe)), k=NUM_TRIES)

    for i in range(len(trial_id)):
        try:
            module = recipe[trial_id[i]]()
            logger.info(f'Loaded module: {module}')
        except Exception as e:
            logger.error(f'Failed to load module: {recipe[trial_id[i]]}')
            logger.error(f'Error: {e}')
            continue
        logger.info(f'Successfully loaded module: {recipe[trial_id[i]]}')

    logger.info(f'Loaded {len(recipe)} modules successfully.')
