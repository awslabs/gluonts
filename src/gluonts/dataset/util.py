# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import itertools
import logging
import random
import os
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    Tuple,
    TypeVar,
    NamedTuple,
)

# Third-party imports
import pandas as pd


T = TypeVar("T")


# Each process has its own copy, so other processes can't interfere
class MPWorkerInfo(object):
    """Contains the current worker information."""

    worker_process = False
    num_workers = 1
    worker_id = 0

    @classmethod
    def set_worker_info(
        cls, num_workers: int, worker_id: int, worker_process: bool
    ):
        cls.num_workers, cls.worker_id, cls.worker_process = (
            num_workers,
            worker_id,
            worker_process,
        )


class DataLoadingBounds(NamedTuple):
    lower: int
    upper: int


def get_bounds_for_mp_data_loading(dataset_len: int) -> DataLoadingBounds:
    """
    Utility function that returns the bounds for which part of the dataset
    should be loaded in this worker.
    """
    if not MPWorkerInfo.worker_process:
        return DataLoadingBounds(0, dataset_len)

    segment_size = int(dataset_len / MPWorkerInfo.num_workers)
    lower = MPWorkerInfo.worker_id * segment_size
    upper = (
        (MPWorkerInfo.worker_id + 1) * segment_size
        if MPWorkerInfo.worker_id + 1 != MPWorkerInfo.num_workers
        else dataset_len
    )
    return DataLoadingBounds(lower=lower, upper=upper)


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
        logging.info(f"Ignoring input file `{ign.name}`.")

    return sorted(chosen)


def to_pandas(instance: dict, freq: str = None) -> pd.Series:
    """
    Transform a dictionary into a pandas.Series object, using its
    "start" and "target" fields.

    Parameters
    ----------
    instance
        Dictionary containing the time series data.
    freq
        Frequency to use in the pandas.Series index.

    Returns
    -------
    pandas.Series
        Pandas time series object.
    """
    target = instance["target"]
    start = instance["start"]
    if not freq:
        freq = start.freqstr
    index = pd.date_range(start=start, periods=len(target), freq=freq)
    return pd.Series(target, index=index)


def take(iterable: Iterable[T], n: int) -> Iterator[T]:
    """Returns up to `n` elements from `iterable`.

    This is similar to xs[:n], except that it works on `Iterable`s and possibly
    consumes the given `iterable`.

    >>> list(take(range(10), 5))
    [0, 1, 2, 3, 4]
    """
    return itertools.islice(iterable, n)


def batcher(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Groups elements from `iterable` into batches of size `batch_size`.

    >>> list(batcher("ABCDEFG", 3))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]

    Unlike the grouper proposed in the documentation of itertools, `batcher`
    doesn't fill up missing values.
    """
    it: Iterator[T] = iter(iterable)

    def get_batch():
        return list(take(it, batch_size))

    # has an empty list so that we have a 2D array for sure
    return iter(get_batch, [])


def dct_reduce(reduce_fn, dcts):
    """Similar to `reduce`, but applies reduce_fn to fields of dicts with the
    same name.

    >>> dct_reduce(sum, [{"a": 1}, {"a": 2}])
    {'a': 3}
    """
    keys = dcts[0].keys()

    return {key: reduce_fn([item[key] for item in dcts]) for key in keys}


def shuffler(stream: Iterable[T], batch_size: int) -> Iterator[T]:
    """Modifies a stream by shuffling items in windows.

    It continously takes `batch_size`-elements from the stream and yields
    elements from each batch  in random order."""

    for batch in batcher(stream, batch_size):
        random.shuffle(batch)
        yield from batch


def cycle(it):
    """Like `itertools.cycle`, but does not store the data."""

    while True:
        yield from it
