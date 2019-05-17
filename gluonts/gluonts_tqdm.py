# Standard library imports
import functools
import sys

# Third-party imports
from tqdm import tqdm as _tqdm

# TODO: when we have upgraded this will give notebook progress bars
# from tqdm.auto import tqdm as _tqdm


@functools.wraps(_tqdm)
def tqdm(*args, **kwargs):
    kwargs = kwargs.copy()
    if not sys.stdout.isatty():
        kwargs.update(mininterval=10.0)

    return _tqdm(*args, **kwargs)
