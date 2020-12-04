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

from collections import OrderedDict
from typing import Optional, Tuple

from mxnet import gluon

from gluonts.core.component import validated
from gluonts.mx import Tensor

from . import LowrankMultivariateGaussian, bijection
from .distribution_output import (
    ArgProj,
    DistributionOutput,
    AffineTransformedDistribution,
)
from .lowrank_multivariate_gaussian import inv_softplus, sigma_minimum


class GPArgProj(gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        rank: int,
        sigma_init: float = 1.0,
        sigma_minimum: float = sigma_minimum,
        mu_ratio: float = 1.0,
        dropout_rate: float = 0.0,
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(prefix=prefix)
        self.param_dim_args = OrderedDict({"mu": 1, "sigma": 1, "w": rank})
        self.mu_ratio = mu_ratio
        self.sigma_init = sigma_init
        self.sigma_minimum = sigma_minimum
        self.W_ratio = 1.0
        self.rank = rank

        def make(name, param_dim):
            net = gluon.nn.HybridSequential()
            net.add(
                gluon.nn.Dense(
                    param_dim,
                    flatten=False,
                    prefix=f"distr_{name}_",
                    weight_initializer="Xavier",
                )
            )
            if dropout_rate > 0:
                net.add(gluon.nn.Dropout(dropout_rate))
            return net

        self.proj = [
            make(name, param_dim)
            for name, param_dim in self.param_dim_args.items()
        ]
        for dense in self.proj:
            self.register_child(dense)

    def hybrid_forward(self, F, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """

        Parameters
        ----------
        F
        x : (..., dim, hidden_dim)

        Returns
        -------
        Returns (mu, D, W) where shapes are (..., dim), (..., dim), (..., dim, rank)
        """

        # TODO 2 concatenate inputs features to x, better names would be great

        # (..., dim)
        mu_vector = self.proj[0](x).squeeze(axis=-1)

        mu = mu_vector * self.mu_ratio

        # (..., dim, rank)
        W_matrix = self.proj[2](x) * self.W_ratio

        # (..., |x| + 1)
        x_plus_w = F.concat(
            x, W_matrix.square().sum(axis=-1, keepdims=True), dim=-1
        )

        # (..., dim)
        D_vector = self.proj[1](x_plus_w).squeeze(axis=-1)

        d_bias = (
            0.0
            if self.sigma_init == 0.0
            else inv_softplus(self.sigma_init ** 2)
        )

        D_positive = (
            F.Activation(D_vector + d_bias, act_type="softrelu")
            + self.sigma_minimum
        )

        return mu, D_positive, W_matrix


class LowrankGPOutput(DistributionOutput):
    @validated()
    def __init__(
        self,
        rank: int,
        dim: Optional[int] = None,  # needed to compute variance
        sigma_init: float = 1.0,
        mu_ratio: float = 1.0,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__(self)
        self.dist_cls = LowrankMultivariateGaussian
        self.dim = dim
        self.rank = rank
        self.args_dim = {"mu": 1, "sigma": 1, "w": rank}
        self.mu_bias = 0.0
        self.sigma_init = sigma_init
        self.mu_ratio = mu_ratio
        self.dropout_rate = dropout_rate

    def get_args_proj(self, prefix: Optional[str] = None) -> ArgProj:

        return GPArgProj(
            rank=self.rank,
            mu_ratio=self.mu_ratio,
            sigma_init=self.sigma_init,
            dropout_rate=self.dropout_rate,
            prefix=prefix,
        )

    def distribution(self, distr_args, loc=None, scale=None, dim=None):
        dist = LowrankMultivariateGaussian(dim, self.rank, *distr_args)
        if loc is None and scale is None:
            return dist
        else:
            return AffineTransformedDistribution(dist, loc=loc, scale=scale)

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)

    @property
    def event_dim(self) -> int:
        return 1
