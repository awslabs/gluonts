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

from gluonts.mx.model.deepvar_hierarchical import (
    constraint_mat,
    coherency_error,
)

TOL = 1e-4

S = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
)

num_bottom_ts = S.shape[1]
A = constraint_mat(S)


@pytest.mark.parametrize(
    "bottom_ts",
    [
        np.random.randint(low=0, high=100, size=num_bottom_ts),  # integer data
        np.random.randint(
            low=1000, high=100000, size=num_bottom_ts
        ),  # large integer data
        np.random.poisson(lam=1, size=num_bottom_ts),
        np.random.negative_binomial(n=1000, p=0.5, size=num_bottom_ts),
        -np.random.negative_binomial(
            n=1000, p=0.5, size=num_bottom_ts
        ),  # negative data
        np.random.standard_normal(size=num_bottom_ts),
    ],
)
def test_coherency_error(bottom_ts):
    all_ts = S @ bottom_ts

    assert coherency_error(mx.nd.array(A), mx.nd.array(all_ts)) < TOL
