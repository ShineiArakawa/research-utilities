"""Multi-processing utilities for parallel execution of functions.
"""

import multiprocessing as mp
import os
import typing

import tqdm.auto


def launch_multi_process(
    func: typing.Callable,
    args: list,
    n_processes: int | None = None,
    verbose: bool = True,
    desc: str = 'Processing ... ',
    context: str = 'spawn',
    is_unordered: bool = False
) -> list:
    """Launch multiple processes to execute a function.

    Parameters
    ----------
    func : typing.Callable
        The function to be executed
    args : list
        The arguments to be passed to the function
    n_processes : int | None, optional
        The number of processes to be launched, by default None (use all available CPUs)
    verbose : bool, optional
        Whether to show the progress bar, by default True
    desc : str, optional
        The description of the progress bar, by default 'Processing ... '
    context : str, optional
        The context of the multiprocessing pool, by default 'spawn'
    is_unordered : bool, optional
        Whether to use unordered processing, by default False
        If True, the results will be returned in the order they are completed.

    Returns
    -------
    list
        The results of the function

    Usage
    -----
    ```python
    def func(arg):
        return arg[0] + arg[1]

    args = [(1, 2), (3, 4), (5, 6)]
    results = launch_multi_process(func, args, n_processes=2)

    print(results)
    # Output: [3, 7, 11]
    ```

    """

    n_processes = n_processes or os.cpu_count()

    with mp.get_context(context).Pool(processes=n_processes) as pool:
        map_fn = pool.imap_unordered if is_unordered else pool.imap

        results = list(
            tqdm.auto.tqdm(
                map_fn(func, args),
                total=len(args),
                desc=desc,
                disable=not verbose
            )
        )

    return results
