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
from functools import partial
from typing import Tuple

# Third-party imports
import numpy as np
import pytest
import mxnet.ndarray as nd
from mxnet import autograd

# First-party imports
from gluonts.mx.distribution.iresnet import iresnet, log_abs_det

RTOL = 1.0e-5
ATOL = 1.0e-8


def allclose(x, y, rtol=RTOL, atol=ATOL):
    return np.allclose(x.asnumpy(), y.asnumpy(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "bijection_func, batch_shape, event_shape",
    [
        (
            partial(
                iresnet,
                num_blocks=3,
                use_caching=True,
                num_hidden_layers=3,
            ),
            (3,),
            (3,),
        ),
        (
            partial(
                iresnet,
                num_blocks=2,
                use_caching=False,
                num_hidden_layers=2,
            ),
            (4,),
            (5,),
        ),
    ],
)
def test_flow_invertibility(
    bijection_func,
    batch_shape,
    event_shape,
):
    input_shape = batch_shape + event_shape
    bijection = bijection_func(event_shape=event_shape)
    bijection.initialize()
    inp = nd.random.randn(*input_shape)
    y_hat = bijection.f(bijection.f_inv(inp))
    x_hat = bijection.f_inv(bijection.f(inp))
    assert allclose(
        inp, y_hat
    ), f"y and y_hat did not match: y = {inp}, y_hat = {y_hat}"
    assert allclose(
        inp, x_hat
    ), f"y and y_hat did not match: x = {inp}, x_hat = {x_hat}"


@pytest.mark.parametrize(
    "bijection_func, batch_shape, event_shape",
    [
        (
            partial(
                iresnet,
                num_blocks=3,
                use_caching=True,
                num_hidden_layers=3,
            ),
            (3,),
            (3,),
        ),
        (
            partial(
                iresnet,
                num_blocks=2,
                use_caching=False,
                num_hidden_layers=2,
            ),
            (4,),
            (5,),
        ),
    ],
)
def test_flow_shapes(
    bijection_func,
    batch_shape,
    event_shape,
):
    bijection = bijection_func(event_shape=event_shape)
    bijection.initialize()
    assert bijection.event_shape == event_shape
    x = nd.zeros(batch_shape + event_shape)
    y = bijection.f(x)
    y_inv = bijection.f_inv(x)
    assert y.shape == batch_shape + event_shape
    assert y_inv.shape == batch_shape + event_shape


def jacobian_autograd(x, y):
    jac = []
    for i in range(y.shape[1]):
        with autograd.record():
            yi = y[:, i]
        dyidx = autograd.grad(yi, [x], create_graph=True)[0]
        jac += [nd.expand_dims(dyidx, 1)]
    return nd.concatenate(jac, 1)


@pytest.mark.parametrize(
    "bijection_func, batch_shape, event_shape",
    [
        (
            partial(
                iresnet,
                num_blocks=3,
                use_caching=True,
                num_hidden_layers=3,
            ),
            (300,),
            (3,),
        ),
        (
            partial(
                iresnet,
                num_blocks=2,
                use_caching=True,
                num_hidden_layers=2,
            ),
            (4,),
            (5,),
        ),
    ],
)
def test_jacobian(bijection_func, batch_shape, event_shape):
    def jacobian(bijection, y):
        y.attach_grad()
        with autograd.record():
            x = bijection.f_inv(y)
        return jacobian_autograd(y, x)

    input_shape = batch_shape + event_shape
    bijection = bijection_func(event_shape=event_shape)
    bijection.initialize()
    inp = nd.random.randn(*input_shape)

    correct_ladj = -log_abs_det(jacobian(bijection, inp))

    # restart outputs, and accumulate intermediate
    # output by calling f_inv, for ladj calculation
    x = bijection.f_inv(inp)
    output_ladj = bijection.log_abs_det_jac(x, inp)
    assert allclose(output_ladj, correct_ladj, atol=1.0e-4, rtol=1.0e-4)
