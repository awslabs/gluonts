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

# Standard library imports
import numpy as np

# Third-party imports
import pytest
import mxnet as mx
from mxnet import autograd

# First-party imports
from gluonts.mx.activation import get_activation, get_activation_deriv


@pytest.mark.parametrize(
    "activation, kwargs",
    [
        ("tanh", dict()),
        ("softrelu", dict()),
        ("elu", dict()),
        ("swish", dict(beta=100)),
        ("swish", dict(beta=0.5)),
        ("lipswish", dict(beta_initializer=mx.init.Constant(1.0))),
        ("lipswish", dict(beta_initializer=mx.init.Constant(10.0))),
    ],
)
def test_activation_deriv(activation, kwargs):
    def get_deriv_autograd(x, act):
        x.attach_grad()
        with autograd.record():
            output = act(x)
        return autograd.grad(output, [x], create_graph=True)[0]

    x = mx.nd.random.randn(500, 20)
    act = get_activation(activation, **kwargs)
    act.initialize()
    correct_deriv = get_deriv_autograd(x, act)
    act_deriv = get_activation_deriv(act)
    output_deriv = act_deriv(mx.ndarray, x)

    assert np.allclose(
        output_deriv.asnumpy(), correct_deriv.asnumpy(), atol=5e-7
    )
