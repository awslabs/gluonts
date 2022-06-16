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

from gluonts.dataset.schema import (
    NumpyArrayField,
    PandasPeriodField,
    Schema,
    FieldWithDefault,
)
from gluonts.exceptions import GluonTSDataError


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
        # using default_value
        (
            {"start": "2022-12-16", "target": [1, 2, 5, 9]},
            {
                "start": PandasPeriodField(freq="D"),
                "target": NumpyArrayField(dtype=float, ndim=1),
                "feat_static_cat": FieldWithDefault(
                    NumpyArrayField(dtype=float, ndim=1), [0.0]
                ),
            },
            {
                "start": pd.Period("2022-12-16", freq="D"),
                "target": np.asarray([1, 2, 5, 9], dtype=float),
                "feat_static_cat": np.asarray([0.0], dtype=float),
            },
        ),
        # not using default_value
        (
            {"start": "2022-12-16", "target": [1, 2], "feat_static_cat": [4]},
            {
                "start": PandasPeriodField(freq="D"),
                "target": NumpyArrayField(dtype=float, ndim=1),
                "feat_static_cat": FieldWithDefault(
                    NumpyArrayField(dtype=float, ndim=1), [0.0]
                ),
            },
            {
                "start": pd.Period("2022-12-16", freq="D"),
                "target": np.asarray([1, 2], dtype=float),
                "feat_static_cat": np.asarray([4], dtype=float),
            },
        ),
        # plain types
        (
            {"x": 13.6, "y": "6", "z": "7.8", "w": 5, "l": (1, 3)},
            {"x": float, "y": int, "z": float, "w": str, "l": list},
            {"x": 13.6, "y": 6, "z": 7.8, "w": "5", "l": [1, 3]},
        ),
        (
            {},
            {
                "x": FieldWithDefault(float, 0),
                "y": FieldWithDefault(int, 0),
                "z": FieldWithDefault(str, ""),
                "w": FieldWithDefault(list, [0.0]),
            },
            {"x": 0, "y": 0.0, "z": "", "w": [0.0]},
        ),
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


@pytest.mark.parametrize(
    "input_data, schema_dir, expected",
    [
        (
            {"start": "2022-12-16", "target": [1, 2, 5, 9]},
            {
                "start": PandasPeriodField(freq="D"),
                "target": NumpyArrayField(dtype=float, ndim=1),
                "feat_static_cat": FieldWithDefault(
                    NumpyArrayField(dtype=float, ndim=1), [0.0]
                ),
            },
            {
                "start": pd.Period("2022-12-16", freq="D"),
                "target": np.asarray([1, 2, 5, 9], dtype=float),
                "feat_static_cat": np.asarray([0.0], dtype=float),
            },
        ),
    ],
)
def test_call_schema_using_default_value(input_data, schema_dir, expected):
    schema = Schema(schema_dir)
    output_data = schema(input_data)
    for field in expected:
        assert field in output_data
        if isinstance(expected[field], np.ndarray):
            assert (output_data[field] == expected[field]).all()
            assert output_data[field].dtype == expected[field].dtype
        else:
            assert output_data[field] == expected[field]


@pytest.mark.parametrize(
    "input_data, schema_dir, expected",
    [
        (
            {"start": "2022-12-16", "target": [1, 2], "feat_static_cat": [4]},
            {
                "start": PandasPeriodField(freq="D"),
                "target": NumpyArrayField(dtype=float, ndim=1),
                "feat_static_cat": FieldWithDefault(
                    NumpyArrayField(dtype=float, ndim=1), [0.0]
                ),
            },
            {
                "start": pd.Period("2022-12-16", freq="D"),
                "target": np.asarray([1, 2], dtype=float),
                "feat_static_cat": np.asarray([4], dtype=float),
            },
        ),
    ],
)
def test_call_schema_not_using_default_value(input_data, schema_dir, expected):
    schema = Schema(schema_dir)
    output_data = schema(input_data)
    for field in expected:
        assert field in output_data
        if isinstance(expected[field], np.ndarray):
            assert (output_data[field] == expected[field]).all()
            assert output_data[field].dtype == expected[field].dtype
        else:
            assert output_data[field] == expected[field]


@pytest.mark.parametrize(
    "input_data, schema_dir, expected",
    [
        (
            {},
            {
                "x": FieldWithDefault(float, 0),
                "y": FieldWithDefault(int, 0),
                "z": FieldWithDefault(str, ""),
                "w": FieldWithDefault(list, [0.0]),
            },
            {"x": 0, "y": 0.0, "z": "", "w": [0.0]},
        ),
    ],
)
def test_call_schema_for_plain_types_using_default_value(
    input_data, schema_dir, expected
):
    schema = Schema(schema_dir)
    output_data = schema(input_data)
    for field in expected:
        assert field in output_data
        assert output_data[field] == expected[field]


@pytest.mark.parametrize(
    "input_data, schema_dir, expected",
    [
        (
            {"x": 2, "y": 5.6, "z": 6, "w": (3, 2), "m": 7.8},
            {
                "x": FieldWithDefault(float, 0),
                "y": FieldWithDefault(int, 0),
                "z": FieldWithDefault(str, ""),
                "w": FieldWithDefault(list, [0.0]),
                "m": int,
            },
            {"x": 2.0, "y": 5, "z": "6", "w": [3, 2], "m": 7},
        ),
    ],
)
def test_call_schema_for_plain_types_not_using_default_value(
    input_data, schema_dir, expected
):
    schema = Schema(schema_dir)
    output_data = schema(input_data)
    for field in expected:
        assert field in output_data
        assert output_data[field] == expected[field]


@pytest.mark.parametrize(
    "input_data, schema_dir, field_name",
    [
        (
            {"start": "2022-12-16", "target": [1, 2, 5, 9]},
            {
                "start": PandasPeriodField(freq="D"),
                "target": NumpyArrayField(dtype=float, ndim=1),
                "feat_static_cat": NumpyArrayField(dtype=float, ndim=1),
            },
            "feat_static_cat",
        ),
    ],
)
def test_call_schema_raise_data_error(input_data, schema_dir, field_name):
    schema = Schema(schema_dir)
    with pytest.raises(GluonTSDataError) as e:
        schema(input_data)
    expected_msg = f"field {field_name} does not occur in the data"
    assert e.value.args[0] == expected_msg
