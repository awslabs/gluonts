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

import pytest
import pandas as pd
import numpy as np

from gluonts.dataset.common import (
    PandasTimestampField,
    NumpyArrayField,
    Schema,
    AnyField,
)


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("B", "2019-11-01"),
        ("W", "2019-11-03"),
        ("M", "2019-11-30"),
        ("12M", "2019-11-30"),
        ("A-DEC", "2019-12-31"),
    ],
)
def test_PandasTimeStampField(freq, expected):
    given = "2019-11-01 12:34:56"
    assert PandasTimestampField._process(given, freq=freq) == pd.Timestamp(
        expected, freq
    )


@pytest.mark.parametrize(
    "array, dtype, dim",
    [
        ([1.0, 2.0, 3, 4, 5], np.int, 1),
        ([[1.0, 2.0, 5], [2.0, 3.0, 4.0]], np.int, 2),
        ([1.0, 2, 3, 4, 5], np.float, 1),
    ],
)
def test_dim_NumpyArrayField(array, dtype, dim):
    proc = NumpyArrayField(is_optional=False, dtype=dtype)
    res = proc(array)
    assert isinstance(res, np.ndarray)
    assert res.dtype == dtype
    assert res.ndim == dim


@pytest.mark.parametrize(
    "entries, expected",
    [
        (
            [{"freq": "5min", "target": [1, 2], "another_field": [1.0, 2.0],}],
            Schema(
                fields={
                    "freq": AnyField(is_optional=False),
                    "target": NumpyArrayField(
                        is_optional=False, dtype=np.int32
                    ),
                    "another_field": NumpyArrayField(
                        is_optional=False, dtype=np.float32
                    ),
                }
            ),
        )
    ],
)
def test_infer_schema(entries, expected):
    inferred = Schema.infer(entries)
    assert inferred == expected


@pytest.mark.parametrize(
    "value, array_type",
    [
        ("a", None),
        (0, None),
        ({"d": 2}, None),
        ("0", None),
        ([1, 2, 3], np.int32),
        ([1, 2.0, 3], np.float32),
        ([1, "NaN", 2], np.float32),
        ([[1,], [2, 3]], None),
    ],
)
def test_compatible_NumpyArrayField(value, array_type):
    int_field = NumpyArrayField(is_optional=False, dtype=np.int32)
    float_field = NumpyArrayField(is_optional=False, dtype=np.float32)
    if array_type is None:
        assert not int_field.is_compatible(value)
        assert not float_field.is_compatible(value)
    if array_type == np.int32:
        assert int_field.is_compatible(value)
        assert float_field.is_compatible(value)
    if array_type == np.float32:
        assert not int_field.is_compatible(value)
        assert float_field.is_compatible(value)
