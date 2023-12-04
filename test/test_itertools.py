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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from numpy.testing import assert_equal
from toolz import take
import pytest

from gluonts.dataset.artificial import constant_dataset
from gluonts.itertools import (
    batcher,
    Cached,
    Chain,
    PickleCached,
    Cyclic,
    IterableSlice,
    SizedIterableSlice,
    PseudoShuffled,
    rows_to_columns,
    columns_to_rows,
    select,
    pluck_attr,
    power_set,
    Map,
    StarMap,
    Filter,
    join_items,
    RandomYield,
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
        (PickleCached(range(4)), (list(range(4)),) * 5),
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


def test_cached_reentry():
    data = Cached(range(10))

    assert len(data) == 10
    assert list(take(5, data)) == list(range(5))
    assert len(data) == 10
    assert list(take(10, data)) == list(range(10))
    assert len(data) == 10


@pytest.mark.parametrize(
    "given, expected",
    [
        ([], {}),
        ([{"a": 1, "b": 2}, {"a": 3, "b": 4}], {"a": [1, 3], "b": [2, 4]}),
    ],
)
@pytest.mark.parametrize("wrapper", [list, np.array])
def test_rows_to_columns(given, expected, wrapper):
    output = rows_to_columns(given, wrapper)
    assert_equal(output, expected)
    assert columns_to_rows(output) == given


def test_sized_iterable_slice():
    def generator():
        for i in range(10):
            yield i

    unsized_iter = generator()
    sized_iter = list(range(10))

    unsized_iter_slice = SizedIterableSlice(unsized_iter, 5)
    with pytest.raises(TypeError):
        len(unsized_iter_slice)

    sized_iter_slice = SizedIterableSlice(sized_iter, 5)
    assert len(sized_iter_slice) == 5

    sized_iter_slice = SizedIterableSlice(sized_iter, 15)
    assert len(sized_iter_slice) == 10


def test_select():
    d = {"a": 1, "b": 2, "c": 3}

    assert select("abc", d) == d
    assert select("ab", d) == {"a": 1, "b": 2}

    assert select("abd", d, ignore_missing=True) == {"a": 1, "b": 2}

    with pytest.raises(KeyError):
        select("abd", d, ignore_missing=False) == {"a": 1, "b": 2}


def test_map():
    data = [1, 2, 3]
    applied = Map(lambda n: n + 1, data)

    assert list(applied) == [2, 3, 4]


def test_starmap():
    def add(a, b, c):
        return a + b + c

    data = [[1, 2, 3], [4, 5, 6]]
    applied = StarMap(add, data)

    assert list(applied) == [add(1, 2, 3), add(4, 5, 6)]


def test_filter():
    data = [1, 2, 3]
    applied = Filter(lambda n: n <= 2, data)

    assert list(applied) == [1, 2]


def test_pluck_attr():
    @dataclass
    class X:
        a: int
        b: int

    xs = [X(1, 2), X(3, 4)]

    assert pluck_attr(xs, "a") == [1, 3]
    assert pluck_attr(xs, "b") == [2, 4]

    with pytest.raises(AttributeError):
        assert pluck_attr(xs, "c")

    assert pluck_attr(xs, "c", 4) == [4, 4]


def test_pluck_attr_curry():
    @dataclass
    class X:
        a: int
        b: int

    xs = [X(1, 2), X(3, 4)]

    get = pluck_attr(xs)

    assert get("a") == [1, 3]
    assert get("b") == [2, 4]

    with pytest.raises(AttributeError):
        assert get("c")

    assert get("c", 4) == [4, 4]


def test_power_set():
    collection = list(range(4))
    subsets = list(power_set(collection))

    expected_subsets = [
        (),
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
        (0, 1, 2, 3),
    ]

    assert len(subsets) == 2 ** len(collection)
    assert len(expected_subsets) == 2 ** len(collection)

    for es in expected_subsets:
        assert es in subsets


def test_join_items():
    left = {"a": 1, "b": 2}
    right = {"a": 3, "c": 4}

    assert list(join_items(left, right)) == [
        ("a", 1, 3),
        ("b", 2, None),
        ("c", None, 4),
    ]

    assert list(join_items(left, right, "inner")) == [
        ("a", 1, 3),
    ]

    assert list(join_items(left, right, "left")) == [
        ("a", 1, 3),
        ("b", 2, None),
    ]

    assert list(join_items(left, right, "right")) == [
        ("a", 1, 3),
        ("c", None, 4),
    ]

    with pytest.raises(Exception):
        oin_items(left, right, "strict")


@pytest.mark.parametrize(
    "iterables, probabilities, sample_size, samples, random_state",
    [
        (
            [[1, 2, 3], [4, 5, 6]],
            [1.0, 0.0],
            5,
            [1, 2, 3],
            np.random.RandomState(0),
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [0.0, 1.0],
            5,
            [4, 5, 6],
            np.random.RandomState(0),
        ),
        (
            [Cyclic([1, 2, 3]), iter([4, 5, 6])],
            [1.0, 0.0],
            5,
            [1, 2, 3, 1, 2],
            np.random.RandomState(0),
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [0.7, 0.3],
            5,
            [1, 4, 2, 3, 5],
            np.random.RandomState(0),
        ),
    ],
)
def test_random_yield(
    iterables, probabilities, sample_size, samples, random_state
):
    it = iter(RandomYield(iterables, probabilities, random_state=random_state))
    generated_samples = list(take(sample_size, it))
    assert generated_samples == samples


@pytest.mark.parametrize(
    "iterables",
    [
        [[1, 2, 3], [4, 5, 6, 7]],
    ],
)
def test_chained(iterables: List[Iterable]):
    expected = list(itertools.chain(*iterables))
    chained = Chain(iterables)

    assert list(chained) == expected
    assert list(chained) == expected
