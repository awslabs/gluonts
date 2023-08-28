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

from functools import partial
from operator import add
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from gluonts.core.component import equals, equals_list
from gluonts.core import serde


class Span(NamedTuple):
    path: Path
    line: int


class BestEpochInfo(NamedTuple):
    params_path: Path
    epoch_no: int
    metric_value: float


# Example Instances
# -----------------

best_epoch_info = BestEpochInfo(
    params_path=Path("foo/bar"), epoch_no=1, metric_value=0.5
)


numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

list_container = [
    best_epoch_info,
    42,
    0.7,
    "fx",
    numpy_array,
]

set_container = {best_epoch_info, 42, 0.7, "fx"}

dict_container = dict(best_epoch_info=best_epoch_info)

simple_types = [
    1,
    42.0,
    "Oh, Romeo",
    np.int32(33),
    np.float64(3.1415),
]  # float('nan')

complex_types = [
    Path("foo/bar"),
    best_epoch_info,
    numpy_array,
]

container_types = [list_container, dict_container, set_container]

examples = simple_types + complex_types + container_types  # type: ignore


def check_equality(expected, actual) -> bool:
    if isinstance(expected, set):
        # Sets are serialized as lists — we check if they have the same elements
        return equals_list(
            sorted(expected, key=hash), sorted(actual, key=hash)
        )
    elif np.issubdtype(type(expected), np.integer):
        # Integer types are expected to be equal exactly
        return np.equal(expected, actual)
    elif np.issubdtype(type(expected), np.inexact):
        # Floating point types are expected to be equal up a certain digit, as specified in np.isclose
        return np.allclose(expected, actual)
    else:
        return equals(expected, actual)


@pytest.mark.parametrize("e", examples)
def test_json_serialization(e) -> None:
    expected, actual = e, serde.load_json(serde.dump_json(e))
    assert check_equality(expected, actual)


def test_timestamp_encode_decode() -> None:
    now = pd.Timestamp.now()
    assert now == serde.decode(serde.encode(now))


def test_string_escape() -> None:
    assert serde.load_json(serde.dump_json(r"a\b")) == r"a\b"


def test_serde_fq():
    add_ = serde.decode(serde.encode(add))
    assert add_(1, 2) == 3

    def foo():
        pass

    with pytest.raises(Exception):
        serde.encode(foo)


def test_serde_partial():
    add_1 = partial(add, 1)

    add_1_ = serde.decode(serde.encode(add_1))

    assert add_1_(2) == 3


class X(serde.Stateless):
    def m(self):
        return 42


def test_serde_method():
    x = X()

    m = serde.decode(serde.encode(x.m))

    assert m() == 42


def test_np_str_dtype():
    a = np.array(["foo"])
    serde.decode(serde.encode(a.dtype)) == a.dtype
