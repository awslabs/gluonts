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
import math
from typing import Optional, Tuple

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import DType, validated
from gluonts.distribution.distribution import (
    Distribution,
    _sample_multiple,
    getF,
)
from gluonts.distribution.distribution_output import DistributionOutput
from gluonts.model.common import Tensor


class MultivariateGaussian(Distribution):
    r"""
    Multivariate Gaussian distribution, specified by the mean vector
    and the Cholesky factor of its covariance matrix.

    Parameters
    ----------
    mu
        mean vector, of shape (..., d)
    L
        Lower triangular Cholesky factor of covariance matrix, of shape
        (..., d, d)
    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet
    """

    is_reparameterizable = True

    def __init__(
        self, mu: Tensor, L: Tensor, F=None, float_type: DType = np.float32
    ) -> None:
        self.mu = mu
        self.F = F if F else getF(mu)
        self.L = L
        self.float_type = float_type

    @property
    def batch_shape(self) -> Tuple:
        return self.mu.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return self.mu.shape[-1:]

    @property
    def event_dim(self) -> int:
        return 1

    def log_prob(self, x: Tensor) -> Tensor:
        # todo add an option to compute loss on diagonal covariance only to save time
        F = self.F

        # remark we compute d from the tensor but we could ask it to the user alternatively
        d = F.ones_like(self.mu).sum(axis=-1).max()

        residual = (x - self.mu).expand_dims(axis=-1)

        # L^{-1} * (x - mu)
        L_inv_times_residual = F.linalg_trsm(self.L, residual)

        ll = (
            F.broadcast_sub(
                -d / 2 * math.log(2 * math.pi), F.linalg_sumlogdiag(self.L)
            )
            - 1
            / 2
            * F.linalg_syrk(L_inv_times_residual, transpose=True).squeeze()
        )

        return ll

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def variance(self) -> Tensor:
        return self.F.linalg_gemm2(self.L, self.L, transpose_b=True)

    def sample_rep(self, num_samples: Optional[int] = None) -> Tensor:
        r"""
        Draw samples from the multivariate Gaussian distributions.
        Internally, Cholesky factorization of the covariance matrix is used:

            sample = L v + mu,

        where L is the Cholesky factor, v is a standard normal sample.

        Parameters
        ----------
        num_samples
            Number of samples to be drawn.
        Returns
        -------
        Tensor
            Tensor with shape (num_samples, ..., d).
        """

        def s(mu: Tensor, L: Tensor) -> Tensor:
            samples_std_normal = self.F.sample_normal(
                mu=self.F.zeros_like(mu),
                sigma=self.F.ones_like(mu),
                dtype=self.float_type,
            ).expand_dims(axis=-1)
            samples = (
                self.F.linalg_gemm2(L, samples_std_normal).squeeze(axis=-1)
                + mu
            )
            return samples

        return _sample_multiple(
            s, mu=self.mu, L=self.L, num_samples=num_samples
        )


class MultivariateGaussianOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: int) -> None:
        self.args_dim = {"mu": dim, "Sigma": dim * dim}
        self.distr_cls = MultivariateGaussian
        self.dim = dim
        self.mask = None

    def lower_triangular_ones(self, F, d: int) -> Tensor:
        mask = F.zeros_like(F.eye(d))
        for k in range(d):
            mask = mask + F.eye(d, d, -k)
        return mask

    def domain_map(self, F, mu_vector, L_vector):
        # apply softplus to the diagonal of L and mask upper coefficient to make it lower-triangular
        # diagonal matrix whose elements are diagonal elements of L mapped through a softplus
        d = self.dim

        # reshape from vector form (..., d * d) to matrix form(..., d, d)
        L_matrix = L_vector.reshape((-2, d, d, -4), reverse=1)

        L_diag = F.broadcast_mul(
            F.Activation(
                F.broadcast_mul(L_matrix, F.eye(d)), act_type='softrelu'
            ),
            F.eye(d),
        )

        mask = self.lower_triangular_ones(F, d)

        L_low = F.broadcast_mul(L_matrix, mask)

        return mu_vector, L_diag + L_low

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
