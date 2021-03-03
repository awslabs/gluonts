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

from typing import List, Optional, Tuple

import numpy as np

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor
from gluonts.mx.util import make_nd_diag

from .distribution import Distribution, _sample_multiple, getF
from .distribution_output import DistributionOutput


class Dirichlet(Distribution):
    r"""
    Dirichlet distribution, specified by the concentration vector alpha of length d.
    https://en.wikipedia.org/wiki/Dirichlet_distribution

    The Dirichlet distribution is defined on the open (d-1)-simplex, which means that
    a sample (or observation) x = (x_0,..., x_{d-1}) must satisfy:

    sum_k x_k = 1 and for all k, x_k > 0.

    Parameters
    ----------
    alpha
        concentration vector, of shape (..., d)

    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, alpha: Tensor, float_type: DType = np.float32) -> None:
        self.alpha = alpha
        self.float_type = float_type

    @property
    def F(self):
        return getF(self.alpha)

    @property
    def args(self) -> List:
        return [self.alpha]

    @property
    def batch_shape(self) -> Tuple:
        return self.alpha.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return self.alpha.shape[-1:]

    @property
    def event_dim(self) -> int:
        return 1

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F

        # Normalize observations in case of mean_scaling
        sum_x = F.sum(x, axis=-1).expand_dims(axis=-1)
        x = F.broadcast_div(x, sum_x)

        alpha = self.alpha

        sum_alpha = F.sum(alpha, axis=-1)
        log_beta = F.sum(F.gammaln(alpha), axis=-1) - F.gammaln(sum_alpha)

        l_x = F.sum((alpha - 1) * F.log(x), axis=-1)
        ll = l_x - log_beta
        return ll

    @property
    def mean(self) -> Tensor:
        F = self.F
        alpha = self.alpha

        sum_alpha = F.sum(alpha, axis=-1)
        return F.broadcast_div(alpha, sum_alpha.expand_dims(axis=-1))

    @property
    def variance(self) -> Tensor:
        F = self.F
        alpha = self.alpha
        d = int(F.ones_like(self.alpha).sum(axis=-1).max().asscalar())

        scale = F.sqrt(F.sum(alpha, axis=-1) + 1).expand_dims(axis=-1)
        scaled_alpha = F.broadcast_div(self.mean, scale)

        cross = F.linalg_gemm2(
            scaled_alpha.expand_dims(axis=-1),
            scaled_alpha.expand_dims(axis=-1),
            transpose_b=True,
        )

        diagonal = make_nd_diag(F, F.broadcast_div(scaled_alpha, scale), d)

        return diagonal - cross

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(alpha: Tensor) -> Tensor:
            F = getF(alpha)
            samples_gamma = F.sample_gamma(
                alpha=alpha, beta=F.ones_like(alpha), dtype=dtype
            )
            sum_gamma = F.sum(samples_gamma, axis=-1, keepdims=True)
            samples_s = F.broadcast_div(samples_gamma, sum_gamma)

            return samples_s

        samples = _sample_multiple(
            s, alpha=self.alpha, num_samples=num_samples
        )

        return samples


class DirichletOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: int) -> None:
        super().__init__(self)
        assert dim > 1, "Dimension should be larger than one."
        self.args_dim = {"alpha": dim}
        self.distr_cls = Dirichlet
        self.dim = dim
        self.mask = None

    def distribution(self, distr_args, loc=None, scale=None) -> Distribution:
        distr = Dirichlet(distr_args)
        return distr

    def domain_map(self, F, alpha_vector):
        # apply softplus to the elements of alpha vector
        alpha = F.Activation(alpha_vector, act_type="softrelu")
        return alpha

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
