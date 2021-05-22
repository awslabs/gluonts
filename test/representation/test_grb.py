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

from gluonts.mx.representation import GlobalRelativeBinning

gr_binning_cases = [
    (
        GlobalRelativeBinning(
            num_bins=6,
            is_quantile=True,
            quantile_scaling_limit=1.0,
        ),
        mx.nd.array(
            [
                [
                    -0.188679,
                    0.377358,
                    0.566038,
                    0.754717,
                    0.943396,
                    1.13208,
                    1.32075,
                    1.50943,
                    1.69811,
                    1.88679,
                ],
                [1.0] * 10,
                [0.857143] * 5 + [1.14286] * 5,
                [1.05263] * 8 + [0.789474] * 2,
                [1.0] * 10,
            ]
        ),
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
                0.334232,
                0.92857149,
                1.0,
                1.03425997,
                1.47765499,
                np.inf,
            ]
        ),
        mx.nd.array(
            [-0.18867899, 0.85714298, 1.0, 1.0, 1.06851995, 1.88679004]
        ),
        mx.nd.array(
            [
                [
                    1,
                    2,
                    2,
                    2,
                    3,
                    5,
                    5,
                    6,
                    6,
                    6,
                ],
                [
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    4,
                    4,
                    4,
                    4,
                    4,
                ],
                [
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    2,
                    2,
                ],
                [
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ],
                [
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                ],
            ]
        ),
    ),
    (
        GlobalRelativeBinning(
            num_bins=8,
            is_quantile=True,
            quantile_scaling_limit=1.0,
        ),
        mx.nd.array(
            [
                [
                    -0.188679,
                    0.377358,
                    0.566038,
                    0.754717,
                    0.943396,
                    1.13208,
                    1.32075,
                    1.50943,
                    1.69811,
                    1.88679,
                ],
                [1.0] * 10,
                [0.857143] * 5 + [1.14286] * 5,
                [1.05263] * 8 + [0.789474] * 2,
                [1.0] * 10,
            ]
        ),
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
                0.334232,
                0.92857149,
                1.0,
                1.0,
                1.02631497,
                1.097745,
                1.51482505,
                np.inf,
            ]
        ),
        mx.nd.array(
            [
                -0.18867899,
                0.85714298,
                1.0,
                1.0,
                1.0,
                1.05262995,
                1.14286005,
                1.88679004,
            ]
        ),
        mx.nd.array(
            [
                [
                    1,
                    2,
                    2,
                    2,
                    3,
                    7,
                    7,
                    7,
                    8,
                    8,
                ],
                [
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    5,
                    5,
                    5,
                    5,
                    5,
                ],
                [
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    2,
                    2,
                ],
                [
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                ],
                [
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                ],
            ]
        ),
    ),
    (
        GlobalRelativeBinning(
            num_bins=6,
            is_quantile=False,
            quantile_scaling_limit=1.0,
        ),
        mx.nd.array(
            [
                [
                    -0.188679,
                    0.377358,
                    0.566038,
                    0.754717,
                    0.943396,
                    1.13208,
                    1.32075,
                    1.50943,
                    1.69811,
                    1.88679,
                ],
                [1.0] * 10,
                [0.857143] * 5 + [1.14286] * 5,
                [1.05263] * 8 + [0.789474] * 2,
                [1.0] * 10,
            ]
        ),
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
        mx.nd.array([-np.inf, -8.0, -4.0, 0.0, 4.0, 8.0, np.inf]),
        mx.nd.array([-10.0, -6.0, -2.0, 2.0, 6.0, 10.0]),
        mx.nd.array(
            [
                [
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                ],
                [
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                ],
                [
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                ],
                [
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                ],
                [
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ],
                [
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                ],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "r, dataset, target, observed, exp_bin_edges, exp_bin_centers, expected_repr",
    gr_binning_cases,
)
def test_gr_binning(
    r, dataset, target, observed, exp_bin_edges, exp_bin_centers, expected_repr
):
    r.initialize_from_array(dataset.asnumpy(), mx.context.cpu())
    target_transf, _, rep_params = r(target, observed, None, [])
    bin_centers_hyb = rep_params[0]
    bin_edges = rep_params[1]

    exp_bin_centers = mx.nd.repeat(
        mx.nd.expand_dims(exp_bin_centers, axis=0),
        len(bin_centers_hyb),
        axis=0,
    )

    assert np.allclose(
        exp_bin_edges.asnumpy(), bin_edges.asnumpy()
    ), f"Bin edges mismatch. Expected: {exp_bin_edges} VS Actual: {bin_edges.asnumpy()}."
    assert np.allclose(
        exp_bin_centers.asnumpy(), bin_centers_hyb.asnumpy()
    ), f"Bin centers mismatch. Expected: {exp_bin_centers} VS Actual: {bin_centers_hyb.asnumpy()}."
    assert np.allclose(
        expected_repr.asnumpy(), target_transf.asnumpy()
    ), f"Representation mismatch. Expected: {expected_repr} VS Actual: {target_transf}."
