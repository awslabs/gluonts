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

import mxnet as mx

from gluonts.distribution.gaussian import Gaussian

DISTR_SHAPE = (3, 4)

DISTR_CASES = [
    Gaussian(
        mu=mx.nd.random.normal(shape=DISTR_SHAPE),
        sigma=mx.nd.random.uniform(shape=DISTR_SHAPE),
    )
]

SLICE_AXIS_CASES = [[(0, 0, None), 3], [(0, 1, 3), 2], [(1, -1, None), 1]]


@pytest.mark.parametrize(
    "slice_axis_args, expected_axis_length", SLICE_AXIS_CASES
)
@pytest.mark.parametrize("distr", DISTR_CASES)
def test_distr_slice_axis(distr, slice_axis_args, expected_axis_length):
    axis, begin, end = slice_axis_args
    distr_sliced = distr.slice_axis(axis, begin, end)

    assert distr_sliced.batch_shape[axis] == expected_axis_length
