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

from gluonts.nursery.anomaly_detection.filters import (
    fill_forward,
    labels_filter,
    n_k_filter,
)

LABELS_FILTER_TEST_CASES = [
    [
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        (3, 3),
    ],
    [
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (3, 2),
    ],
    [
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        (3, 2),
    ],
    [
        [0, 1, 0, 1, np.nan, np.nan, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        (3, 2),
    ],
    [
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        (1, 1),
    ],
]


@pytest.mark.parametrize(
    "input, expected_output, params", LABELS_FILTER_TEST_CASES
)
def test_labels_filter(input, expected_output, params):
    out = labels_filter(
        np.array(input),
        *params,
        forward_fill=True,
    )
    assert np.allclose(out, expected_output)


@pytest.mark.parametrize(
    "input, expected_output, params", LABELS_FILTER_TEST_CASES
)
def test_n_k_filter_defaults(input, expected_output, params):
    out = n_k_filter(
        np.array(input),
        *params,
        forward_fill=True,
    )
    assert np.allclose(out, expected_output)


NK_FILTER_TEST_CASES = [
    [
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        (3, 3, 2, 2),
    ],
    [
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        (5, 3, 3, 1),
    ],
    [
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        (4, 4, 2, 2),
    ],
    [
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        (5, 4, 3),
    ],
    [
        [0, 1, np.nan, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        (5, 4, 3),
    ],
    [
        [0, 1, 0, 1, np.nan, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        (5, 4, 3),
    ],
]


@pytest.mark.parametrize(
    "input, expected_output, params", NK_FILTER_TEST_CASES
)
def test_n_k_filter_defaults_custom(input, expected_output, params):
    out = n_k_filter(
        np.array(input),
        *params,
        forward_fill=True,
    )
    assert np.allclose(out, expected_output)


@pytest.mark.parametrize(
    "input, output",
    [
        [
            [1, 1, 1, 3, np.nan, np.nan],
            [1, 1, 1, 3, 3, 3],
        ],
        [
            [2, np.nan, 1, 3, 4, np.nan],
            [2, 2, 1, 3, 4, 4],
        ],
        [
            [np.nan, np.nan, 1, 3, 4, np.nan],
            [np.nan, np.nan, 1, 3, 4, 4],
        ],
    ],
)
def test_forward_fill(input, output):
    assert np.allclose(
        output,
        fill_forward(input),
        equal_nan=True,
    )
