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
    target = instance['target']
    start = instance['start']
    if not freq:
        freq = start.freqstr
    index = pd.date_range(start=start, periods=len(target), freq=freq)
    return pd.Series(target, index=index)
