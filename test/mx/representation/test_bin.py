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

import mxnet as mx
import numpy as np
import pytest

from gluonts.mx.representation import CustomBinning

binning_cases = [
    (
        CustomBinning(bin_centers=np.linspace(-1, 10, 5)),
        mx.nd.array(
            [
                [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [3.0] * 10,
                [0.0] * 5 + [3.0] * 5,
                [2.0] * 8 + [1.5] * 2,
                [0.0] * 10,
                [1.0] * 10,
            ]
        ),
        mx.nd.array(
            [
                [1.0] * 10,
                [1.0] * 10,
                [0.0] * 5 + [1.0] * 5,
                [1.0] * 9 + [1.0] * 1,
                [0.0] * 10,
                [1.0] * 10,
            ]
        ),
        mx.nd.array([-np.inf, 0.375, 3.125, 5.875, 8.625, np.inf]),
        mx.nd.array(
            [
                [
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                ],
                [
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
                [
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                [
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
            ]
        ),
    ),
    (
        CustomBinning(bin_centers=np.linspace(-10, 10, 8)),
        mx.nd.array(
            [
                [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [3.0] * 10,
                [0.0] * 5 + [3.0] * 5,
                [2.0] * 8 + [1.5] * 2,
                [0.0] * 10,
                [1.0] * 10,
            ]
        ),
        mx.nd.array(
            [
                [1.0] * 10,
                [1.0] * 10,
                [0.0] * 5 + [1.0] * 5,
                [1.0] * 9 + [1.0] * 1,
                [0.0] * 10,
                [1.0] * 10,
            ]
        ),
        mx.nd.array(
            [
                -np.inf,
                -8.57142857,
                -5.71428571,
                -2.85714286,
                0.0,
                2.85714286,
                5.71428571,
                8.57142857,
                np.inf,
            ]
        ),
        mx.nd.array(
            [
                [
                    4.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    8.0,
                    8.0,
                ],
                [
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                ],
                [
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                ],
                [
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                ],
                [
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                ],
                [
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                ],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "r, target, observed, exp_bin_edges, expected_repr",
    binning_cases,
)
def test_binning(r, target, observed, exp_bin_edges, expected_repr):
    r.initialize_from_array(np.array([]), mx.context.cpu())
    target_transf, _, rep_params = r(target, observed, None, [])
    bin_edges = rep_params[1]

    assert np.allclose(
        exp_bin_edges.asnumpy(), bin_edges.asnumpy()
    ), f"Bin edges mismatch. Expected: {exp_bin_edges} VS Actual: {bin_edges.asnumpy()}."
    assert np.allclose(
        expected_repr.asnumpy(), target_transf.asnumpy()
    ), f"Representation mismatch. Expected: {expected_repr} VS Actual: {target_transf}."
