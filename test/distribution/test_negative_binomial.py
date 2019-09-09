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
from gluonts.distribution.neg_binomial import NegativeBinomialOutput


def test_issue_287():
    network_output = mx.nd.ones(shape=(10,))
    distr_output = NegativeBinomialOutput()
    args_proj = distr_output.get_args_proj()
    args_proj.initialize(init=mx.init.Constant(-1e2))
    distr_args = args_proj(network_output)
    distr = distr_output.distribution(distr_args)
    x = mx.nd.array([1.0])
    ll = distr.log_prob(x)
    assert np.isfinite(ll.asnumpy())
