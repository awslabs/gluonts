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
import pytest
from numpy.testing import assert_equal

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


def test_call_schema_using_default_value():
    schema = Schema(
        {
            "start": PandasPeriodField(freq="D"),
            "target": NumpyArrayField(dtype=float, ndim=1),
            "feat_static_cat": FieldWithDefault(
                NumpyArrayField(dtype=float, ndim=1), [0.0]
            ),
        }
    )
    output_data = schema({"start": "2022-12-16", "target": [1, 2, 5, 9]})

    assert output_data["start"] == pd.Period("2022-12-16", freq="D")
    assert_equal(output_data["target"], np.asarray([1, 2, 5, 9], dtype=float))
    assert_equal(
        output_data["feat_static_cat"], np.asarray([0.0], dtype=float)
    )


def test_call_schema_not_using_default_value():
    schema = Schema(
        {
            "start": PandasPeriodField(freq="D"),
            "target": NumpyArrayField(dtype=int, ndim=1),
            "feat_static_cat": FieldWithDefault(
                NumpyArrayField(dtype=int, ndim=1), [0.0]
            ),
        }
    )
    output_data = schema(
        {"start": "2022-12-16", "target": [1.3, 2.5], "feat_static_cat": [6.7]}
    )

    assert output_data["start"] == pd.Period("2022-12-16", freq="D")
    assert_equal(output_data["target"], np.asarray([1, 2], dtype=int))
    assert_equal(output_data["feat_static_cat"], np.asarray([6], dtype=int))


def test_call_schema_for_plain_types_using_default_value():
    schema = Schema(
        {
            "x": FieldWithDefault(float, 0),
            "y": FieldWithDefault(int, 0),
            "z": FieldWithDefault(str, ""),
            "w": FieldWithDefault(list, [0.0]),
        }
    )
    output_data = schema({})

    assert output_data["x"] == 0
    assert output_data["y"] == 0.0
    assert output_data["z"] == ""
    assert output_data["w"] == [0.0]


def test_call_schema_for_plain_types_not_using_default_value():
    schema = Schema(
        {
            "x": FieldWithDefault(float, 0),
            "y": FieldWithDefault(int, 0),
            "z": FieldWithDefault(str, ""),
            "w": FieldWithDefault(list, [0.0]),
            "m": int,
        }
    )
    output_data = schema({"x": 2, "y": 5.6, "z": 6, "w": (3, 2), "m": 7.8})

    assert output_data["x"] == 2.0
    assert output_data["y"] == 5
    assert output_data["z"] == "6"
    assert output_data["w"] == [3, 2]
    assert output_data["m"] == 7


def test_call_schema_raise_data_error():
    schema = Schema(
        {
            "start": PandasPeriodField(freq="D"),
            "target": NumpyArrayField(dtype=float, ndim=1),
            "feat_static_cat": NumpyArrayField(dtype=float, ndim=1),
        }
    )

    with pytest.raises(GluonTSDataError) as e:
        schema({"start": "2022-12-16", "target": [1, 2, 5, 9]})
    expected_msg = "field feat_static_cat does not occur in the data"
    assert e.value.args[0] == expected_msg
