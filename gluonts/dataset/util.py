# Standard library imports
import logging
import os
from pathlib import Path
from typing import Callable, Iterator, List, Tuple, TypeVar

# Third-party imports
import pandas as pd


T = TypeVar('T')


def _split(
    it: Iterator[T], fn: Callable[[T], bool]
) -> Tuple[List[T], List[T]]:
    left, right = [], []

    for val in it:
        if fn(val):
            left.append(val)
        else:
            right.append(val)

    return left, right


def _list_files(directory: Path) -> Iterator[Path]:
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            yield Path(dirname, filename)


def true_predicate(*args) -> bool:
    return True


def find_files(
    data_dir: Path, predicate: Callable[[Path], bool] = true_predicate
) -> List[Path]:
    all_files = _list_files(data_dir)
    chosen, ignored = _split(all_files, predicate)

    for ign in ignored:
        logging.info(f'Ignoring input file `{ign.name}`.')

    return sorted(chosen)


def to_pandas(instance: dict, freq: str = None) -> pd.Series:
    target = instance['target']
    start = instance['start']
    if not freq:
        freq = start.freqstr
    index = pd.DatetimeIndex(start=start, periods=len(target), freq=freq)
    return pd.Series(target, index=index)
