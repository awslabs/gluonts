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

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from gluonts.dataset.schema import NumpyArrayField, PandasPeriodField, Schema


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
def test_call_PandasPeriodField(freq, expected):
    field_type = PandasPeriodField(freq=freq)
    given = "2019-11-01 12:34:56"
    assert field_type(given) == pd.Period(expected, freq)


@pytest.mark.parametrize(
    "array, dtype, dim",
    [
        ([1.0, 2.0, 3, 4, 5], int, 1),
        ([[1.0, 2.0, 5], [2.0, 3.0, 4.0]], int, 2),
        ([1, 2, 3, 4, 5], float, 1),
        (pa.array([1, 2, 3, 4, 5]), float, 1),
    ],
)
def test_call_NumpyArrayField(array, dtype, dim):
    field_type = NumpyArrayField(dtype=dtype, ndim=dim)
    np_array = field_type(array)
    assert isinstance(np_array, np.ndarray)
    assert np_array.dtype == dtype
    assert np_array.ndim == dim


@pytest.mark.parametrize(
    "value, array_type",
    [
        ("a", None),
        (0, None),
        ({"d": 2}, None),
        ("0", None),
        ([[1], [2, 3]], None),
        ([1, 2, 3], int),
        ([1, 2.0, 3], float),
        ([1, "NaN", 2], float),
        (pa.array([1, 3, 5]), int),
    ],
)
def test_compatible_NumpyArrayField(value, array_type):
    int_field = NumpyArrayField(dtype=int)
    float_field = NumpyArrayField(dtype=float)
    if array_type is None:
        assert not int_field.is_compatible(value)
        assert not float_field.is_compatible(value)
    if array_type == int:
        assert int_field.is_compatible(value)
        assert float_field.is_compatible(value)
    if array_type == float:
        assert not int_field.is_compatible(value)
        assert float_field.is_compatible(value)


@pytest.mark.parametrize(
    "input_data, schema_dir, expected",
    [
        (
            {"start": "2022-12-16", "target": [1, 2, 5, 9]},
            {
                "start": PandasPeriodField(freq="D"),
                "target": NumpyArrayField(dtype=float, ndim=1),
                "feat_static_cat": NumpyArrayField(dtype=float, ndim=1),
            },
            {
                "start": pd.Period("2022-12-16", freq="D"),
                "target": np.asarray([1, 2, 5, 9], dtype=float),
                "feat_static_cat": np.asarray([0.0], dtype=float),
            },
        )
    ],
)
def test_call_schema(input_data, schema_dir, expected):
    schema = Schema(schema_dir)
    output_data = schema(input_data)
    for field in expected:
        assert field in output_data
        if isinstance(expected[field], np.ndarray):
            assert (output_data[field] == expected[field]).all()
            assert output_data[field].dtype == expected[field].dtype
        else:
            assert output_data[field] == expected[field]
