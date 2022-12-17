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
from mxnet import nd

from gluonts.mx.block.encoder import HierarchicalCausalConv1DEncoder


@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("hybridize", [True, False])
def test_hierarchical_cnn_encoders(use_residual, hybridize) -> None:
    num_ts = 2
    ts_len = 10
    num_static_feat = 2
    num_dynamic_feat = 5

    test_data = nd.arange(num_ts * ts_len).reshape(shape=(num_ts, ts_len, 1))
    test_static_feat = nd.random.randn(num_ts, num_static_feat)
    test_dynamic_feat = nd.random.randn(num_ts, ts_len, num_dynamic_feat)

    chl_dim = [30, 30, 30]
    ks_seq = [3] * len(chl_dim)
    dial_seq = [1, 3, 9]

    cnn = HierarchicalCausalConv1DEncoder(
        dial_seq,
        ks_seq,
        chl_dim,
        use_residual,
        use_dynamic_feat=True,
        use_static_feat=True,
    )
    cnn.collect_params().initialize()

    if hybridize:
        cnn.hybridize()

    true_shape = (num_ts, ts_len, 31) if use_residual else (num_ts, ts_len, 30)

    assert (
        cnn(test_data, test_static_feat, test_dynamic_feat)[1].shape
        == true_shape
    )
