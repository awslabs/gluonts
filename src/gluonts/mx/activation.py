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
from typing import Callable

# Third-party imports
import mxnet as mx
import mxnet.gluon.nn as nn

# First-party imports
from gluonts.core.component import validated
from gluonts.mx import Tensor


def get_activation(activation: str, **kwargs) -> nn.HybridBlock:
    """

    Parameters
    ----------
    activation
        Activation type

    Returns
    -------
    mxnet.gluon.HybridBlock
        Activation object

    """
    if activation in ["relu", "sigmoid", "softrelu", "softsign", "tanh"]:
        return nn.Activation(activation=activation, **kwargs)
    if activation == "lrelu":
        return nn.LeakyReLU(alpha=0.2, **kwargs)
    if activation == "elu":
        return nn.ELU(**kwargs)
    if activation == "swish":
        return nn.Swish(**kwargs)
    if activation == "lipswish":
        return LipSwish(**kwargs)
    raise NotImplementedError(activation)


def get_activation_deriv(act: nn.HybridBlock) -> Callable:
    """

    Parameters
    ----------
    act
        Activation object
    Returns
    -------
    Callable
        Derivative function of associated activation

    """
    if isinstance(act, nn.Activation):
        activation = act._act_type
        if activation == "relu":
            raise NotImplementedError(activation)
        if activation == "sigmoid":
            raise NotImplementedError(activation)
        if activation == "tanh":
            return deriv_tanh
        if activation == "softrelu":
            return deriv_softrelu
        if activation == "softsign":
            raise NotImplementedError(activation)
    if isinstance(act, nn.ELU):
        return partial(deriv_elu, alpha=act._alpha)
    if isinstance(act, nn.Swish):
        return partial(deriv_swish, beta=act._beta)
    if isinstance(act, LipSwish):
        return partial(deriv_lipswish, beta=act.params.get("beta").data())
    raise NotImplementedError(
        f'No derivative function for activation "' f'{act.__class__.__name__}"'
    )


def deriv_tanh(F, x: Tensor) -> Tensor:
    """
    Derivative function of Tanh activation computed at point `x`.

    Parameters
    ----------
    F
        A module that can either refer to the Symbol API or the NDArray API in MXNet.

    x
        Input tensor

    Returns
    -------
    Tensor
        Derivative tensor

    """
    return 1 - F.tanh(x) ** 2


def deriv_softrelu(F, x: Tensor) -> Tensor:
    """
    Derivative function of SoftRelu activation computed at point `x`.

    Parameters
    ----------
    F
        A module that can either refer to the Symbol API or the NDArray API in MXNet.
    x
        Input tensor

    Returns
    -------
    Tensor
        Derivative tensor

    """
    e = mx.nd.exp(x)
    return e / (1 + e)


def deriv_elu(F, x: Tensor, alpha: float = 1.0) -> Tensor:
    """
    Derivative function of Elu activation computed at point `x`.

    Parameters
    ----------
    F
        A module that can either refer to the Symbol API or the NDArray API in MXNet.
    x
        Input tensor
    alpha
        alpha parameter of Elu

    Returns
    -------
    Tensor
        Derivative tensor

    """
    m = x > 0
    return m + (1 - m) * (F.LeakyReLU(x, act_type="elu", slope=alpha) + alpha)


def deriv_swish(F, x: Tensor, beta: Tensor) -> Tensor:
    """
    Derivative function of Swish activation computed at point `x`.

    Parameters
    ----------
    F
        A module that can either refer to the Symbol API or the NDArray API in MXNet.
    x
        Input tensor
    beta
        beta parameter of LipSwish

    Returns
    -------
    Tensor
        Derivative tensor

    """
    f = x * F.sigmoid(beta * x, name="fwd")
    return beta * f + F.sigmoid(beta * x) * (1 - beta * f)


def deriv_lipswish(F, x: Tensor, beta: Tensor) -> Tensor:
    """
    Derivative function of LipSwish activation computed at point `x`.
    Parameters
    ----------
    F
        A module that can either refer to the Symbol API or the NDArray API in MXNet.
    x
        Input tensor
    beta
        beta parameter in LipSwish activation
    Returns
    -------
    Tensor
        Derivative tensor

    """
    return deriv_swish(F, x, beta) / 1.1


class LipSwish(nn.HybridBlock):
    """
    Implemented LipSwish activation, i.e. LipSwish(z) := Swish(z)/ 1.1 with a learnable parameter beta.
    """

    @validated()
    def __init__(
        self,
        beta_initializer: mx.init.Initializer = mx.init.Constant(1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        with self.name_scope():
            self.beta = self.params.get(
                "beta", shape=(1,), init=beta_initializer
            )

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x: Tensor, beta: Tensor) -> Tensor:
        """

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        x
            Input tensor
        beta
            beta parameter of activation

        Returns
        -------
        Tensor
            output of forward

        """
        return x * F.sigmoid(beta * x, name="fwd") / 1.1
