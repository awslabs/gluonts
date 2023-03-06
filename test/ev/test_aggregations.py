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

import pytest

from gluonts.ev import Mean, Sum
from gluonts.itertools import power_set

VALUE_STREAM = [
    [
        np.full((3, 5), np.nan),
        np.full((3, 5), np.nan),
        np.full((3, 5), np.nan),
    ],
    [
        np.array([[0, np.nan], [0, 0]]),
        np.array([[0, 5], [-5, np.nan]]),
    ],
    [
        np.full(shape=(3, 3), fill_value=1),
        np.full(shape=(1, 3), fill_value=4),
    ],
]

SUM_RES_AXIS_NONE = [
    0,
    0,
    21,
]

SUM_RES_AXIS_0 = [
    np.zeros(5),
    np.array([-5, 5]),
    np.array([7, 7, 7]),
]
SUM_RES_AXIS_1 = [
    np.zeros(9),
    np.array([0, 0, 5, -5]),
    np.array([3, 3, 3, 12]),
]


MEAN_RES_AXIS_NONE = [
    np.nan,
    0,
    1.75,
]

MEAN_RES_AXIS_0 = [
    np.full(5, np.nan),
    np.array([-1.25, 2.5]),
    np.array([1.75, 1.75, 1.75]),
]
MEAN_RES_AXIS_1 = [
    np.full(9, np.nan),
    np.array([0, 0, 2.5, -5]),
    np.array([1, 1, 1, 4]),
]


@pytest.mark.parametrize(
    "value_stream, res_axis_none, res_axis_0, res_axis_1",
    zip(VALUE_STREAM, SUM_RES_AXIS_NONE, SUM_RES_AXIS_0, SUM_RES_AXIS_1),
)
def test_Sum(value_stream, res_axis_none, res_axis_0, res_axis_1):
    for axis, expected_result in zip(
        [None, 0, 1], [res_axis_none, res_axis_0, res_axis_1]
    ):
        sum = Sum(axis=axis)
        for values in value_stream:
            sum.step(np.ma.masked_invalid(values))

        np.testing.assert_almost_equal(sum.get(), expected_result)


@pytest.mark.parametrize(
    "value_stream, res_axis_none, res_axis_0, res_axis_1",
    zip(VALUE_STREAM, MEAN_RES_AXIS_NONE, MEAN_RES_AXIS_0, MEAN_RES_AXIS_1),
)
def test_Mean(value_stream, res_axis_none, res_axis_0, res_axis_1):
    for axis, expected_result in zip(
        [None, 0, 1], [res_axis_none, res_axis_0, res_axis_1]
    ):
        mean = Mean(axis=axis)
        for values in value_stream:
            mean.step(np.ma.masked_invalid(values))

        np.testing.assert_almost_equal(mean.get(), expected_result)


@pytest.mark.parametrize("shape", [(4, 9, 5, 2)])
@pytest.mark.parametrize("axis", [None] + list(power_set([0, 1, 2, 3])))
def test_high_dim(shape: tuple, axis):
    batch_count = 3

    value_stream = [np.random.random(shape) for _ in range(batch_count)]
    all_values = np.concatenate(value_stream)

    sum = Sum(axis=axis)
    for values in value_stream:
        sum.step(values)

    actual_sum = sum.get()
    expected_sum = all_values.sum(axis=axis)
    np.testing.assert_almost_equal(actual_sum, expected_sum)

    mean = Mean(axis=axis)
    for values in value_stream:
        mean.step(values)

    actual_mean = mean.get()
    expected_mean = all_values.mean(axis=axis)
    np.testing.assert_almost_equal(actual_mean, expected_mean)
