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

from gluonts.mx.representation import (
    CustomBinning,
    DimExpansion,
    HybridRepresentation,
    LocalAbsoluteBinning,
    RepresentationChain,
)

hyb_cases = [
    (
        HybridRepresentation(
            representations=[
                RepresentationChain(
                    chain=[
                        CustomBinning(bin_centers=np.linspace(-1, 10, 5)),
                        DimExpansion(),
                    ]
                ),
                RepresentationChain(
                    chain=[
                        CustomBinning(bin_centers=np.linspace(-10, 10, 8)),
                        DimExpansion(),
                    ]
                ),
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
        [
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
        ],
    ),
    (
        HybridRepresentation(
            representations=[
                RepresentationChain(
                    chain=[
                        CustomBinning(bin_centers=np.linspace(-1, 10, 5)),
                        DimExpansion(),
                    ]
                ),
                RepresentationChain(
                    chain=[
                        LocalAbsoluteBinning(num_bins=6, is_quantile=True),
                        DimExpansion(),
                    ]
                ),
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
        [
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
        ],
    ),
    (
        HybridRepresentation(
            representations=[
                RepresentationChain(
                    chain=[
                        LocalAbsoluteBinning(num_bins=6, is_quantile=True),
                        DimExpansion(),
                    ]
                ),
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
        [
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
        ],
    ),
]


@pytest.mark.parametrize(
    "r, target, observed, expected_repr",
    hyb_cases,
)
def test_hyb(r, target, observed, expected_repr):
    r.initialize_from_array(np.array([]), mx.context.cpu())
    target_transf, _, _ = r(target, observed, None, [])

    for i in range(len(expected_repr)):
        exp_loc = expected_repr[i].asnumpy()
        target_loc = target_transf[:, :, i].asnumpy()
        print(target_loc)
        assert np.allclose(
            exp_loc, target_loc
        ), f"Representation mismatch. Expected: {exp_loc} VS Actual: {target_loc}."
