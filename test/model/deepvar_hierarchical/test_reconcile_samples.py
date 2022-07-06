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
    null_space_projection_mat,
    reconcile_samples,
    coherency_error,
)

TOL = 1e-2

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
reconciliation_mat = null_space_projection_mat(A)


@pytest.mark.parametrize(
    "samples",
    [
        np.random.randint(low=10000, high=100000, size=(10, 32, S.shape[0])),
        100.0 + np.random.poisson(lam=1, size=(10, 32, S.shape[0])),
        np.random.negative_binomial(n=1000, p=0.5, size=(10, 32, S.shape[0])),
        -np.random.negative_binomial(
            n=1000, p=0.5, size=(10, 32, S.shape[0])
        ),  # negative data
        100.0 + 2.0 * np.random.standard_normal(size=(10, 32, S.shape[0])),
    ],
)
@pytest.mark.parametrize(
    "seq_axis",
    [
        None,
        [0],
        [1],
        [0, 1],
        [1, 0],
    ],
)
def test_reconciliation_error(samples, seq_axis):
    coherent_samples = reconcile_samples(
        reconciliation_mat=mx.nd.array(reconciliation_mat),
        samples=mx.nd.array(samples),
        seq_axis=seq_axis,
    )

    assert coherency_error(mx.nd.array(A), coherent_samples) < TOL
