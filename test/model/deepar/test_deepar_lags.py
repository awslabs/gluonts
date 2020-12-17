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

import itertools

import mxnet as mx

from gluonts.model.deepar._network import DeepARTrainingNetwork


def test_lagged_subsequences():
    N = 8
    T = 96
    C = 2
    lags = [1, 2, 3, 24, 48]
    I = len(lags)
    sequence = mx.nd.random.normal(shape=(N, T, C))
    S = 48

    # (batch_size, sub_seq_len, target_dim, num_lags)
    lagged_subsequences = DeepARTrainingNetwork.get_lagged_subsequences(
        F=mx.nd,
        sequence=sequence,
        sequence_length=sequence.shape[1],
        indices=lags,
        subsequences_length=S,
    )

    assert (N, S, C, I) == lagged_subsequences.shape

    # checks that lags value behave as described as in the get_lagged_subsequences contract
    for i, j, k in itertools.product(range(N), range(S), range(I)):
        assert (
            (
                lagged_subsequences[i, j, :, k]
                == sequence[i, -lags[k] - S + j, :]
            )
            .asnumpy()
            .all()
        )
