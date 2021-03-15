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

import itertools
import pickle
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest

from gluonts.dataset.artificial import constant_dataset
from gluonts.itertools import (
    batcher,
    Cached,
    Cyclic,
    IterableSlice,
    PseudoShuffled,
)


@pytest.mark.parametrize(
    "data, n, expected", [([1, 2, 3], 7, [1, 2, 3, 1, 2, 3, 1]), ([], 4, [])]
)
def test_cyclic(data: Iterable, n: int, expected: List) -> None:
    cyclic_data = Cyclic(data)
    actual = list(itertools.islice(cyclic_data, n))
    assert actual == expected


@pytest.mark.parametrize(
    "data",
    [
        range(20),
        constant_dataset()[1],
    ],
)
def test_pseudo_shuffled(data: Iterable) -> None:
    list_data = list(data)
    shuffled_iter = PseudoShuffled(iter(list_data), shuffle_buffer_length=5)
    shuffled_data = list(shuffled_iter)
    assert len(shuffled_data) == len(list_data)
    assert all(d in shuffled_data for d in list_data)


@pytest.mark.parametrize(
    "data, expected_elements_per_iteration",
    [
        (Cached(range(4)), (list(range(4)),) * 5),
        (batcher(range(10), 3), ([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]], [])),
        (IterableSlice(range(10), 3), ([0, 1, 2],) * 5),
        (
            IterableSlice(iter(range(10)), 3),
            ([0, 1, 2], [3, 4, 5], [6, 7, 8], [9], []),
        ),
        (
            IterableSlice(iter(Cyclic(range(5))), 3),
            ([0, 1, 2], [3, 4, 0], [1, 2, 3], [4, 0, 1]),
        ),
    ],
)
def test_iterate_multiple_times(
    data: Iterable, expected_elements_per_iteration: Tuple[List]
):
    for expected_elements in expected_elements_per_iteration:
        assert list(data) == expected_elements


@pytest.mark.parametrize(
    "iterable, assert_content",
    [
        (Cached(range(5)), True),
        (PseudoShuffled(range(20), 5), False),
        (IterableSlice(Cyclic(range(5)), 9), True),
    ],
)
def test_pickle(iterable: Iterable, assert_content: bool):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(Path(tmpdir) / "temp.pickle", "wb") as fp:
            pickle.dump(iterable, fp)

        with open(Path(tmpdir) / "temp.pickle", "rb") as fp:
            iterable_copy = pickle.load(fp)

    data = list(iterable)
    data_copy = list(iterable_copy)

    assert len(data) == len(data_copy)

    if assert_content:
        assert data == data_copy
