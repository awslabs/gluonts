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
from typing import Optional

# Third-party imports
import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import init

# First-party imports
from gluonts.core.component import validated
from gluonts.mx.context import get_mxnet_context
from gluonts.mx import Tensor
from gluonts.mx.activation import get_activation


EPSILON = 1e-12


class SNDense(mx.gluon.HybridBlock):
    """
    Dense layer with spectral normalization applied to
    weights, as in [BJC19]_.
    """

    @validated()
    def __init__(
        self,
        units: int,
        in_units: int,
        coeff: float = 0.9,
        activation: Optional[str] = None,
        use_bias: bool = True,
        flatten: bool = True,
        weight_initializer: init.Initializer = init.Orthogonal(scale=0.9),
        bias_initializer="zeros",
        dtype="float32",
        num_power_iter: int = 1,
        ctx: Optional[mx.Context] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._coeff = coeff
        self._flatten = flatten
        self._ctx = ctx if ctx is not None else get_mxnet_context()
        self._num_power_iter = num_power_iter
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            self._weight = self.params.get(
                "weight",
                shape=(units, in_units),
                init=weight_initializer,
                dtype=dtype,
            )
            self._u = self.params.get(
                "u", init=mx.init.Normal(), shape=(1, units)
            )

            if use_bias:
                self._bias = self.params.get(
                    "bias", shape=(units,), init=bias_initializer, dtype=dtype
                )
            else:
                self._bias = None

            if activation is not None:
                self._act = get_activation(activation, prefix=activation + "_")
            else:
                self._act = None

    @property
    def weight(self):
        return self._spectral_norm(
            self._weight.data(self._ctx), self._u.data(self._ctx)
        )

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x, _weight, _u, _bias=None):
        """

        Parameters
        ----------
        x
            Input Tensor of SNDense layer

        Returns
        -------
        Tensor
            Output Tensor of SNDense layer

        """
        act = nd.FullyConnected(
            data=x,
            weight=self._spectral_norm(_weight, _u),
            bias=_bias,
            no_bias=_bias is None,
            num_hidden=self._units,
            flatten=self._flatten,
            name="fwd",
        )

        if self._act is not None:
            act = self._act(act)

        return act

    def __repr__(self):
        s = "{name}({layout}, {act})"
        shape = self._weight.shape
        return s.format(
            name=self.__class__.__name__,
            act=self._act if self._act else "linear",
            layout="{0} -> {1}".format(
                shape[1] if shape[1] else None, shape[0]
            ),
        )

    def _spectral_norm(self, weight: Tensor, u: Tensor) -> Tensor:
        """
        Adapted from https://github.com/apache/incubator-mxnet/blob/master/example/gluon/sn_gan/model.py
        """
        w = weight
        w_mat = nd.reshape(w, [w.shape[0], -1])

        _u = u
        _v = None

        for _ in range(self._num_power_iter):
            _v = nd.L2Normalization(nd.dot(_u, w_mat))
            _u = nd.L2Normalization(nd.dot(_v, w_mat.T))

        sigma = nd.sum(nd.dot(_u, w_mat) * _v)

        # this is different from standard spectral normalization
        sigma = nd.maximum(nd.ones(1, ctx=self._ctx), sigma / self._coeff)

        if sigma == 0.0:
            sigma = EPSILON

        with autograd.pause():
            self._u.set_data(_u)

        return w / sigma
