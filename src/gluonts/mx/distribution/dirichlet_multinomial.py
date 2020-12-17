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

from typing import Optional, Tuple

import numpy as np

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor
from gluonts.mx.util import make_nd_diag

from .distribution import Distribution, _sample_multiple, getF
from .distribution_output import DistributionOutput


class DirichletMultinomial(Distribution):
    r"""
    Dirichlet-Multinomial distribution, specified by the concentration vector alpha of length dim, and a number of
    trials n_trials.
    https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution

    The Dirichlet-Multinomial distribution is a discrete multivariate probability distribution, a sample
    (or observation) x = (x_0,..., x_{dim-1}) must satisfy:

    sum_k x_k = n_trials and for all k, x_k is a non-negative integer.

    Such a sample can be obtained by first drawing a vector p from a Dirichlet(alpha) distribution, then x is
    drawn from a Multinomial(p) with n trials

    Parameters
    ----------
    dim
        Dimension of any sample

    n_trials
        Number of trials

    alpha
        concentration vector, of shape (..., dim)

    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet
    """

    is_reparameterizable = False

    @validated()
    def __init__(
        self,
        dim: int,
        n_trials: int,
        alpha: Tensor,
        float_type: DType = np.float32,
    ) -> None:
        self.dim = dim
        self.n_trials = n_trials
        self.alpha = alpha
        self.float_type = float_type

    @property
    def F(self):
        return getF(self.alpha)

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
        n_trials = self.n_trials
        alpha = self.alpha

        sum_alpha = F.sum(alpha, axis=-1)

        ll = (
            F.gammaln(sum_alpha)
            + F.gammaln(F.ones_like(sum_alpha) * (n_trials + 1.0))
            - F.gammaln(sum_alpha + n_trials)
        )

        beta_matrix = (
            F.gammaln(x + alpha) - F.gammaln(x + 1) - F.gammaln(alpha)
        )

        ll = ll + F.sum(beta_matrix, axis=-1)

        return ll

    @property
    def mean(self) -> Tensor:
        F = self.F
        alpha = self.alpha
        n_trials = self.n_trials

        sum_alpha = F.sum(alpha, axis=-1)
        return (
            F.broadcast_div(alpha, sum_alpha.expand_dims(axis=-1)) * n_trials
        )

    @property
    def variance(self) -> Tensor:
        F = self.F
        alpha = self.alpha
        d = self.dim
        n_trials = self.n_trials

        sum_alpha = F.sum(alpha, axis=-1)
        scale = F.sqrt(
            (sum_alpha + 1) / (sum_alpha + n_trials) / n_trials
        ).expand_dims(axis=-1)
        scaled_alpha = F.broadcast_div(self.mean / n_trials, scale)

        cross = F.linalg_gemm2(
            scaled_alpha.expand_dims(axis=-1),
            scaled_alpha.expand_dims(axis=-1),
            transpose_b=True,
        )

        diagonal = make_nd_diag(F, F.broadcast_div(scaled_alpha, scale), d)

        dir_variance = diagonal - cross

        return dir_variance

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        dim = self.dim
        n_trials = self.n_trials

        def s(alpha: Tensor) -> Tensor:
            F = getF(alpha)
            samples_gamma = F.sample_gamma(
                alpha=alpha, beta=F.ones_like(alpha), dtype=dtype
            )
            sum_gamma = F.sum(samples_gamma, axis=-1, keepdims=True)
            samples_s = F.broadcast_div(samples_gamma, sum_gamma)

            cat_samples = F.sample_multinomial(samples_s, shape=n_trials)
            return F.sum(F.one_hot(cat_samples, dim), axis=-2)

        samples = _sample_multiple(
            s, alpha=self.alpha, num_samples=num_samples
        )

        return samples


class DirichletMultinomialOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: int, n_trials: int) -> None:
        super().__init__(self)
        assert dim > 1, "Dimension must be larger than one."
        self.dim = dim
        self.n_trials = n_trials
        self.args_dim = {"alpha": dim}
        self.distr_cls = DirichletMultinomial
        self.dim = dim
        self.mask = None

    def distribution(self, distr_args, loc=None, scale=None) -> Distribution:
        distr = DirichletMultinomial(self.dim, self.n_trials, distr_args)
        return distr

    def domain_map(self, F, alpha_vector):
        # apply softplus to the elements of alpha vector
        alpha = F.Activation(alpha_vector, act_type="softrelu")
        return alpha

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
