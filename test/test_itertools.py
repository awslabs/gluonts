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
from typing import Iterable, List

import pytest

from gluonts.dataset.artificial import constant_dataset
from gluonts.itertools import cyclic, pseudo_shuffled


@pytest.mark.parametrize(
    "data, n, expected", [([1, 2, 3], 7, [1, 2, 3, 1, 2, 3, 1]), ([], 4, [])]
)
def test_cyclic(data: Iterable, n: int, expected: List) -> None:
    cyclic_data = cyclic(data)
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
    shuffled_iter = pseudo_shuffled(iter(list_data), shuffle_buffer_length=5)
    shuffled_data = list(shuffled_iter)
    assert len(shuffled_data) == len(list_data)
    assert all(d in shuffled_data for d in list_data)
