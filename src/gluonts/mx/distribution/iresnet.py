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

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.util import _broadcast_param
from gluonts.mx.block.snmlp import SNMLPBlock
from gluonts.mx.distribution.distribution import getF
from gluonts.mx.distribution.bijection import (
    BijectionHybridBlock,
    ComposedBijectionHybridBlock,
)


def log_abs_det(A: Tensor) -> Tensor:
    """
    Logarithm of the absolute value of matrix `A`
    Parameters
    ----------
    A
        Tensor matrix from which to compute the log absolute value of its determinant

    Returns
    -------
        Tensor

    """
    F = getF(A)
    A_squared = F.linalg.gemm2(A, A, transpose_a=True)
    L = F.linalg.potrf(A_squared)
    return F.diag(L, axis1=-2, axis2=-1).abs().log().sum(-1)


class InvertibleResnetHybridBlock(BijectionHybridBlock):
    """
    Based on [BJC19]_,
    apart from f and f_inv that are swapped.
    """

    @validated()
    def __init__(
        self,
        event_shape,
        hidden_units: int = 16,
        num_hidden_layers: int = 1,
        num_inv_iters: int = 10,
        ignore_logdet: bool = False,
        activation: str = "lipswish",
        num_power_iter: int = 1,
        flatten: bool = False,
        coeff: float = 0.9,
        use_caching: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert len(event_shape) == 1
        self._event_shape = event_shape
        self._hidden_units = hidden_units
        self._num_hidden_layers = num_hidden_layers
        self._num_inv_iters = num_inv_iters
        self._ignore_logdet = ignore_logdet
        self._activation = activation
        self._coeff = coeff
        self._num_power_iter = num_power_iter
        self._flatten = flatten
        self._use_caching = use_caching
        self._cached_x = None
        self._cached_y = None

        with self.name_scope():
            self._block = SNMLPBlock(
                self._event_shape[0],
                self._hidden_units,
                self._event_shape[0],
                num_hidden_layers=self._num_hidden_layers,
                activation=self._activation,
                flatten=self._flatten,
            )

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def event_dim(self) -> int:
        return len(self.event_shape)

    def f(self, x: Tensor) -> Tensor:
        """
        Forward transformation of iResnet

        Parameters
        ----------
        x
            observations

        Returns
        -------
        Tensor
            transformed tensor `\text{iResnet}(x)`

        """
        if x is self._cached_x:
            return self._cached_y
        y = x
        for _ in range(self._num_inv_iters):
            y = x - self._block(y)
        if self._use_caching:
            self._cached_x = x
            self._cached_y = y
        return y

    def f_inv(self, y: Tensor) -> Tensor:
        """
        Inverse transformation of `iResnet`

        Parameters
        ----------
        y
            input tensor
        Returns
        -------
        Tensor
            transformed tensor `\text{iResnet}^{-1}(y)`

        """
        if y is self._cached_y:
            return self._cached_x
        x = y + self._block(y)
        if self._use_caching:
            self._cached_x = x
            self._cached_y = y
        return x

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithm of the absolute value of the Jacobian determinant corresponding to the iResnet Transform

        Parameters
        ----------
        x
            input of the forward transformation or output of the inverse transform
        y
            output of the forward transform or input of the inverse transform

        Returns
        -------
        Tensor
            Jacobian evaluated for x as input or y as output
        """
        if self._ignore_logdet:
            assert x is not None
            ladj = mx.nd.zeros(x.shape[0])
        else:
            # we take the negative value since we use the forward pass of mlp to
            # compute the inverse, and we want ladj of forward, which is
            # opposite of ladj of reverse
            jac_block = self._block.jacobian(y)
            batch_shape, (output_dim, input_dim) = (
                jac_block.shape[:-2],
                jac_block.shape[-2:],
            )
            identity = _broadcast_param(
                mx.nd.eye(output_dim, input_dim),
                axes=range(len(batch_shape)),
                sizes=batch_shape,
            )
            ladj = -log_abs_det(identity + jac_block)

        return ladj


def iresnet(num_blocks: int, **block_kwargs) -> ComposedBijectionHybridBlock:
    """

    Parameters
    ----------
    num_blocks
        number of iResnet blocks
    block_kwargs
        keyword arguments given to initialize each block object

    Returns
    -------

    """
    return ComposedBijectionHybridBlock(
        [
            InvertibleResnetHybridBlock(**block_kwargs)
            for _ in range(num_blocks)
        ]
    )
