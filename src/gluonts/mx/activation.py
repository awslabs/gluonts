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

# Third-party imports
import mxnet as mx
import mxnet.gluon.nn as nn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


def get_activation(name, *args, **kwargs):
    if name in ["relu", "sigmoid", "softrelu", "softsign", "tanh"]:
        return partial(nn.Activation, activation=name, *args, **kwargs)
    if name == "lrelu":
        return partial(nn.LeakyReLU, alpha=0.2, *args, **kwargs)
    if name == "elu":
        return nn.ELU
    if name == "swish":
        return partial(nn.Swish, *args, **kwargs)
    if name == "lipswish":
        return partial(LipSwish, *args, **kwargs)
    raise NotImplementedError(name)


def get_activation_deriv(act):
    if isinstance(act, nn.Activation):
        name = act._act_type
        if name == "relu":
            raise NotImplementedError(name)
        if name == "sigmoid":
            raise NotImplementedError(name)
        if name == "tanh":
            return deriv_tanh
        if name == "softrelu":
            return deriv_softrelu
        if name == "softsign":
            raise NotImplementedError(name)
    if isinstance(act, nn.ELU):
        return partial(deriv_elu, alpha=act._alpha)
    if isinstance(act, nn.Swish):
        return partial(deriv_swish, beta=act._beta)
    if isinstance(act, LipSwish):
        return partial(deriv_lipswish, beta=act.params.get("beta").data())
    raise NotImplementedError(
        f'No derivative function for activation "' f'{act.__class__.__name__}"'
    )


def deriv_tanh(F, x):
    return 1 - F.tanh(x) ** 2


def deriv_softrelu(F, x):
    e = mx.nd.exp(x)
    return e / (1 + e)


def deriv_elu(F, x, alpha=1.0):
    m = x > 0
    return m + (1 - m) * (F.LeakyReLU(x, act_type="elu", slope=alpha) + alpha)


def deriv_swish(F, x, beta):
    f = x * F.sigmoid(beta * x, name="fwd")
    return beta * f + F.sigmoid(beta * x) * (1 - beta * f)


def deriv_lipswish(F, x, beta):
    return deriv_swish(F, x, beta) / 1.1


class LipSwish(nn.HybridBlock):
    @validated()
    def __init__(self, beta_initializer=mx.init.Constant(1.0), **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.beta = self.params.get(
                "beta", shape=(1,), init=beta_initializer
            )

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x: Tensor, beta: Tensor) -> Tensor:
        return x * F.sigmoid(beta * x, name="fwd") / 1.1
