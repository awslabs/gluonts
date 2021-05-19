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

from gluonts.model.renewal._predictor import (
    DeepRenewalProcessSampleOutputTransform,
)


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            [[[[3, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]]]],
            [[[0, 0, 3, 5, 0, 4, 0]]],
        ),
        (
            [[[[7, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]]]],
            [[[0, 0, 0, 0, 0, 0, 3]]],
        ),
        (
            [[[[1, 9, 2, 3, 1, 1, 1], [14, 5, 4, 1, 1, 1, 1]]]],
            [[[14, 0, 0, 0, 0, 0, 0]]],
        ),
        (
            [[[[8, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]]]],
            [[[0, 0, 0, 0, 0, 0, 0]]],
        ),
        (
            [
                [
                    [[3, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]],
                    [[3, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]],
                ]
            ],
            [[[0, 0, 3, 5, 0, 4, 0], [0, 0, 3, 5, 0, 4, 0]]],
        ),
        (
            [
                [[[3, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]]],
                [[[3, 2, 1, 1, 1, 1, 1], [6, 7, 8, 9, 1, 1, 1]]],
            ],
            [[[0, 0, 3, 5, 0, 4, 0]], [[0, 0, 6, 0, 7, 8, 9]]],
        ),
    ],
)
def test_output_transform(input, expected):
    expected = np.array(expected)
    tf = DeepRenewalProcessSampleOutputTransform()
    out = tf({}, np.array(input))

    assert np.allclose(out, expected)
    assert out.shape == expected.shape
