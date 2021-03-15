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

from gluonts.mx.representation import LocalAbsoluteBinning

la_binning_cases = [
    (
        LocalAbsoluteBinning(num_bins=6, is_quantile=True),
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
                [
                    -np.inf,
                    0.9,
                    3.7,
                    5.5,
                    7.3,
                    9.1,
                    np.inf,
                ],
                [
                    -np.inf,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    np.inf,
                ],
                [
                    -np.inf,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    np.inf,
                ],
                [
                    -np.inf,
                    1.7,
                    1.95,
                    2.0,
                    2.0,
                    2.0,
                    np.inf,
                ],
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [
                    -np.inf,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    np.inf,
                ],
            ]
        ),
        mx.nd.array(
            [
                [
                    -1.0,
                    2.8,
                    4.6,
                    6.4,
                    8.2,
                    10.0,
                ],
                [
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                [
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                [
                    1.5,
                    1.9,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
            ]
        ),
        mx.nd.array(
            [
                [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            ]
        ),
    ),
    (
        LocalAbsoluteBinning(num_bins=6, is_quantile=False),
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
                [
                    -np.inf,
                    0.1,
                    2.3,
                    4.5,
                    6.7,
                    8.9,
                    np.inf,
                ],
                [
                    -np.inf,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    np.inf,
                ],
                [
                    -np.inf,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    np.inf,
                ],
                [
                    -np.inf,
                    1.55,
                    1.65,
                    1.75,
                    1.85,
                    1.95,
                    np.inf,
                ],
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [
                    -np.inf,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    np.inf,
                ],
            ]
        ),
        mx.nd.array(
            [
                [
                    -1.0,
                    1.2,
                    3.4,
                    5.6,
                    7.8,
                    10.0,
                ],
                [
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                [
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                [
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.9,
                    2.0,
                ],
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
            ]
        ),
        mx.nd.array(
            [
                [1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "r, target, observed, exp_bin_edges, exp_bin_centers, expected_repr",
    la_binning_cases,
)
def test_la_binning(
    r, target, observed, exp_bin_edges, exp_bin_centers, expected_repr
):
    target_transf, _, rep_params = r(target, observed, None, [])
    bin_centers_hyb = rep_params[0].asnumpy()
    bin_edges_hyb = rep_params[1].asnumpy()

    assert np.allclose(
        exp_bin_edges.asnumpy(), bin_edges_hyb
    ), f"Bin edges mismatch. Expected: {exp_bin_edges} VS Actual: {bin_edges_hyb}."
    assert np.allclose(
        exp_bin_centers.asnumpy(), bin_centers_hyb
    ), f"Bin centers mismatch. Expected: {exp_bin_centers} VS Actual: {bin_centers_hyb}."
    assert np.allclose(
        expected_repr.asnumpy(), target_transf.asnumpy()
    ), f"Representation mismatch. Expected: {expected_repr} VS Actual: {target_transf}."
