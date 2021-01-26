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
from typing import List, Tuple
import warnings

# Third-party imports
import mxnet as mx
from mxnet.ndarray import linalg_gemm2 as gemm2

# First-party imports
from gluonts.mx import Tensor
from gluonts.core.component import validated
from gluonts.mx.util import _broadcast_param
from gluonts.mx.activation import get_activation, get_activation_deriv
from gluonts.mx.block.sndense import SNDense


def jacobian_sn_mlp_block_bf(
    layers: List[Tuple[mx.gluon.HybridBlock, Tensor]]
) -> Tensor:
    """
    Brute force computation of the jacobian of a SNMlpBlock
    jac is of shape (Batch dim1, ..., Output dim, Input dim)

    Parameters
    ----------
    layers
        A list of tuples where each tuple (layer, input) is associated to a composing layer of the SNMLPBlock, where layer corresponds to the associated object layer, along with its input tensor.

    Returns
    -------
    Tensor
        Jacobian of the SNMLPBlock computed at a given input

    """
    for i, (layer, input) in enumerate(layers[::-1]):
        if isinstance(layer, SNDense):
            # broadcast weight of size (Output dim, Input dim)
            # to (Batch dim1, ..., Output dim, Input dim)
            jac_t = _broadcast_param(
                layer.weight,
                axes=range(len(input.shape[:-1])),
                sizes=input.shape[:-1],
            )
            if i == 0:
                jac = jac_t
            else:
                jac = gemm2(jac, jac_t)
        else:
            # act_deriv is of shape (Batch dim1, ..., Input dim)
            act_deriv = get_activation_deriv(layer)(mx.ndarray, input)
            # to (Batch dim1, ..., Output dim, Input dim) to fit the jacobian
            jac_t = act_deriv.expand_dims(len(jac.shape[:-2])).broadcast_axes(
                axis=len(jac.shape[:-2]), size=jac.shape[-2]
            )
            jac = jac * jac_t
    return jac


class SNMLPBlock(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        in_units: int,
        hidden_units: int,
        out_units: int,
        num_hidden_layers: int = 2,
        activation: str = "lipswish",
        jacobian_method: str = "bf",
        num_power_iter: int = 1,
        coeff: float = 0.9,
        flatten: bool = False,
    ):
        super().__init__()
        self._in_units = in_units
        self._hidden_units = hidden_units
        self._out_units = out_units
        self._num_hidden_layers = num_hidden_layers
        self._activation = activation
        self._jacobian_method = jacobian_method
        self._num_power_iter = num_power_iter
        self._coeff = coeff
        self._weight_initializer = mx.init.Orthogonal(scale=self._coeff)
        self._bias_initializer = "zeros"
        self._flatten = flatten
        self._cached_inputs: List[Tensor] = []

        in_dim = self._in_units
        with self.name_scope():
            self._layers: List[mx.gluon.HybridBlock] = []
            for i in range(self._num_hidden_layers):
                lin = SNDense(
                    self._hidden_units,
                    in_units=in_dim,
                    activation=None,
                    num_power_iter=self._num_power_iter,
                    weight_initializer=self._weight_initializer,
                    bias_initializer=self._bias_initializer,
                    coeff=self._coeff,
                    flatten=self._flatten,
                )
                act = get_activation(
                    self._activation, prefix=self._activation + str(i)
                )
                in_dim = self._hidden_units
                self.register_child(lin)
                self.register_child(act)
                self._layers += [lin, act]

            last_lin = SNDense(
                self._out_units,
                in_units=in_dim,
                activation=None,
                num_power_iter=self._num_power_iter,
                weight_initializer=self._weight_initializer,
                bias_initializer=self._bias_initializer,
                coeff=self._coeff,
                flatten=self._flatten,
            )
            self.register_child(last_lin)
            self._layers += [last_lin]

    def get_weights(self):
        return [
            layer.weight
            for layer in self._layers
            if isinstance(layer, SNDense)
        ]

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x
            Input Tensor

        Returns
        -------
        Tensor
            output of SNMLPBlock

        """
        self._cached_inputs = []
        for layer in self._layers:
            self._cached_inputs += [x]
            x = layer(x)
        return x

    def jacobian(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x
            Input Tensor

        Returns
        -------
        Tensor
            Jacobian of the SNMLPBlock evaluated at x.

        """
        if self._jacobian_method == "ignore":
            return x * 0
        elif self._jacobian_method == "bf":
            if (
                len(self._cached_inputs) > 0
                and self._cached_inputs[0] is not x
            ):
                warnings.warn(
                    "Input not the same, recomputing forward for jacobian term..."
                )
                self(x)
            return jacobian_sn_mlp_block_bf(
                [(l, i) for l, i in zip(self._layers, self._cached_inputs)]
            )
        raise NotImplementedError(self._jacobian_method)
