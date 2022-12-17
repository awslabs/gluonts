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

import math
from typing import Optional, Tuple

import numpy as np
from mxnet import gluon

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import Distribution, _sample_multiple, getF
from .distribution_output import (
    ArgProj,
    DistributionOutput,
    AffineTransformedDistribution,
)

sigma_minimum = 1e-3


def capacitance_tril(F, rank: Tensor, W: Tensor, D: Tensor) -> Tensor:
    r"""

    Parameters
    ----------
    F
    rank
    W : (..., dim, rank)
    D : (..., dim)

    Returns
    -------
        the capacitance matrix :math:`I + W^T D^{-1} W`

    """
    # (..., dim, rank)
    Wt_D_inv_t = F.broadcast_div(W, D.expand_dims(axis=-1))

    # (..., rank, rank)
    K = F.linalg_gemm2(Wt_D_inv_t, W, transpose_a=True)

    # (..., rank, rank)
    Id = F.broadcast_mul(F.ones_like(K), F.eye(rank))

    # (..., rank, rank)
    return F.linalg.potrf(K + Id)


def log_det(F, batch_D: Tensor, batch_capacitance_tril: Tensor) -> Tensor:
    r"""
    Uses the matrix determinant lemma.

    .. math::
        \log|D + W W^T| = \log|C| + \log|D|,

    where :math:`C` is the capacitance matrix :math:`I + W^T D^{-1} W`, to
    compute the log determinant.

    Parameters
    ----------
    F
    batch_D
    batch_capacitance_tril

    Returns
    -------

    """
    log_D = batch_D.log().sum(axis=-1)
    log_C = 2 * F.linalg.sumlogdiag(batch_capacitance_tril)
    return log_C + log_D


def mahalanobis_distance(
    F, W: Tensor, D: Tensor, capacitance_tril: Tensor, x: Tensor
) -> Tensor:
    r"""
    Uses the Woodbury matrix identity

    .. math::
        (W W^T + D)^{-1} = D^{-1} - D^{-1} W C^{-1} W^T D^{-1},

    where :math:`C` is the capacitance matrix :math:`I + W^T D^{-1} W`, to
    compute the squared Mahalanobis distance :math:`x^T (W W^T + D)^{-1} x`.

    Parameters
    ----------
    F
    W
        (..., dim, rank)
    D
        (..., dim)
    capacitance_tril
        (..., rank, rank)
    x
        (..., dim)

    Returns
    -------

    """
    xx = x.expand_dims(axis=-1)

    # (..., rank, 1)
    Wt_Dinv_x = F.linalg_gemm2(
        F.broadcast_div(W, D.expand_dims(axis=-1)), xx, transpose_a=True
    )

    # compute x^T D^-1 x, (...,)
    maholanobis_D_inv = F.broadcast_div(x.square(), D).sum(axis=-1)

    # (..., rank)
    L_inv_Wt_Dinv_x = F.linalg_trsm(capacitance_tril, Wt_Dinv_x).squeeze(
        axis=-1
    )

    maholanobis_L = L_inv_Wt_Dinv_x.square().sum(axis=-1).squeeze()

    return F.broadcast_minus(maholanobis_D_inv, maholanobis_L)


def lowrank_log_likelihood(
    rank: int, mu: Tensor, D: Tensor, W: Tensor, x: Tensor
) -> Tensor:

    F = getF(mu)

    dim = F.ones_like(mu).sum(axis=-1).max()

    dim_factor = dim * math.log(2 * math.pi)

    if W is not None:
        batch_capacitance_tril = capacitance_tril(F=F, rank=rank, W=W, D=D)

        log_det_factor = log_det(
            F=F, batch_D=D, batch_capacitance_tril=batch_capacitance_tril
        )

        mahalanobis_factor = mahalanobis_distance(
            F=F, W=W, D=D, capacitance_tril=batch_capacitance_tril, x=x - mu
        )
    else:
        log_det_factor = D.log().sum(axis=-1)
        x_centered = x - mu
        mahalanobis_factor = F.broadcast_div(x_centered.square(), D).sum(
            axis=-1
        )

    ll: Tensor = -0.5 * (
        F.broadcast_add(dim_factor, log_det_factor) + mahalanobis_factor
    )

    return ll


class LowrankMultivariateGaussian(Distribution):
    r"""
    Multivariate Gaussian distribution, with covariance matrix parametrized
    as the sum of a diagonal matrix and a low-rank matrix

    .. math::
        \Sigma = D + W W^T

    When `W = None` the covariance matrix is just diagonal.

    The implementation is strongly inspired from Pytorch:
    https://github.com/pytorch/pytorch/blob/master/torch/distributions/lowrank_multivariate_normal.py.

    Complexity to compute log_prob is :math:`O(dim * rank + rank^3)` per element.

    Parameters
    ----------
    dim
        Dimension of the distribution's support
    rank
        Rank of W
    mu
        Mean tensor, of shape (..., dim)
    D
        Diagonal term in the covariance matrix, of shape (..., dim)
    W
        Low-rank factor in the covariance matrix, of shape (..., dim, rank)
        Optional; if not provided, the covariance matrix is just diagonal.
    """  # noqa: E501

    is_reparameterizable = True

    @validated()
    def __init__(
        self,
        dim: int,
        rank: int,
        mu: Tensor,
        D: Tensor,
        W: Optional[Tensor] = None,
    ) -> None:
        self.dim = dim
        self.rank = rank
        self.mu = mu
        self.D = D
        self.W = W
        self.Cov = None

    @property
    def F(self):
        return getF(self.mu)

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
        return lowrank_log_likelihood(
            rank=self.rank, mu=self.mu, D=self.D, W=self.W, x=x
        )

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def variance(self) -> Tensor:
        assert self.dim is not None
        F = self.F

        if self.Cov is not None:
            return self.Cov
        # reshape to a matrix form (..., d, d)
        D_matrix = self.D.expand_dims(-1) * F.eye(self.dim)

        if self.W is not None:
            W_matrix = F.linalg_gemm2(self.W, self.W, transpose_b=True)
            self.Cov = D_matrix + W_matrix
        else:
            self.Cov = D_matrix

        return self.Cov

    def sample_rep(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        r"""
        Draw samples from the multivariate Gaussian distribution:

        .. math::
            s = \mu + D u + W v,

        where :math:`u` and :math:`v` are standard normal samples.

        Parameters
        ----------
        num_samples
            number of samples to be drawn.
        dtype
            Data-type of the samples.

        Returns
        -------
            tensor with shape (num_samples, ..., dim)
        """

        def s(mu: Tensor, D: Tensor, W: Optional[Tensor] = None) -> Tensor:
            F = getF(mu)

            samples_D = F.sample_normal(
                mu=F.zeros_like(mu), sigma=F.ones_like(mu), dtype=dtype
            )
            cov_D = D.sqrt() * samples_D

            if W is not None:
                # dummy only use to get the shape (..., rank, 1)
                dummy_tensor = F.linalg_gemm2(
                    W, mu.expand_dims(axis=-1), transpose_a=True
                ).squeeze(axis=-1)

                samples_W = F.sample_normal(
                    mu=F.zeros_like(dummy_tensor),
                    sigma=F.ones_like(dummy_tensor),
                    dtype=dtype,
                )

                cov_W = F.linalg_gemm2(
                    W, samples_W.expand_dims(axis=-1)
                ).squeeze(axis=-1)

                samples = mu + cov_D + cov_W
            else:
                samples = mu + cov_D

            return samples

        return _sample_multiple(
            s, mu=self.mu, D=self.D, W=self.W, num_samples=num_samples
        )

    @classmethod
    def fit(cls, F, samples: Tensor, rank: int = 0) -> Distribution:
        """
        Returns an instance of `LowrankMultivariateGaussian` after fitting
        parameters to the given data. Only the special case of `rank` = 0 is
        supported at the moment.

        Parameters
        ----------
        F
        samples
            Tensor of shape (num_samples, batch_size, seq_len, target_dim)
        rank
            Rank of W

        Returns
        -------
        Distribution instance of type `LowrankMultivariateGaussian`.

        """
        # TODO: Implement it for the general case: `rank` > 0
        assert rank == 0, "Fit is not only implemented for the case rank = 0!"

        # Compute mean and variances
        mu = samples.mean(axis=0)
        var = F.square(samples - samples.mean(axis=0)).mean(axis=0)
        return cls(dim=samples.shape[-1], rank=rank, mu=mu, D=var)


def inv_softplus(y):
    if y < 20.0:
        # y = log(1 + exp(x))  ==>  x = log(exp(y) - 1)
        return np.log(np.exp(y) - 1)
    else:
        return y


class LowrankMultivariateGaussianOutput(DistributionOutput):
    @validated()
    def __init__(
        self,
        dim: int,
        rank: int,
        sigma_init: float = 1.0,
        sigma_minimum: float = sigma_minimum,
    ) -> None:
        super().__init__(self)
        self.distr_cls = LowrankMultivariateGaussian
        self.dim = dim
        self.rank = rank
        if rank == 0:
            self.args_dim = {"mu": dim, "D": dim}
        else:
            self.args_dim = {"mu": dim, "D": dim, "W": dim * rank}
        self.mu_bias = 0.0
        self.sigma_init = sigma_init
        self.sigma_minimum = sigma_minimum

    def get_args_proj(self, prefix: Optional[str] = None) -> ArgProj:
        return ArgProj(
            args_dim=self.args_dim,
            domain_map=gluon.nn.HybridLambda(self.domain_map),
            prefix=prefix,
        )

    def distribution(
        self, distr_args, loc=None, scale=None, **kwargs
    ) -> Distribution:
        distr = LowrankMultivariateGaussian(self.dim, self.rank, *distr_args)
        if loc is None and scale is None:
            return distr
        else:
            return AffineTransformedDistribution(distr, loc=loc, scale=scale)

    def domain_map(self, F, mu_vector, D_vector, W_vector=None):
        r"""

        Parameters
        ----------
        F
        mu_vector
            Tensor of shape (..., dim)
        D_vector
            Tensor of shape (..., dim)
        W_vector
            Tensor of shape (..., dim * rank )

        Returns
        -------
        Tuple
            A tuple containing tensors mu, D, and W, with shapes
            (..., dim), (..., dim), and (..., dim, rank), respectively.

        """

        d_bias = (
            inv_softplus(self.sigma_init**2)
            if self.sigma_init > 0.0
            else 0.0
        )

        # sigma_minimum helps avoiding cholesky problems, we could also jitter
        # However, this seems to cause the maximum likelihood estimation to
        # take longer to converge. This needs to be re-evaluated.
        D_diag = (
            F.Activation(D_vector + d_bias, act_type="softrelu")
            + self.sigma_minimum**2
        )

        if self.rank == 0:
            return mu_vector + self.mu_bias, D_diag
        else:
            assert (
                W_vector is not None
            ), "W_vector cannot be None if rank is not zero!"
            # reshape from vector form (..., d * rank) to matrix form
            # (..., d, rank)
            W_matrix = W_vector.reshape(
                (-2, self.dim, self.rank, -4), reverse=1
            )
            return mu_vector + self.mu_bias, D_diag, W_matrix

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
