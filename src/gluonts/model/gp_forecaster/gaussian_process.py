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
from typing import List, Optional, Tuple

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import DType
from gluonts.distribution import MultivariateGaussian
from gluonts.distribution.distribution import getF
from gluonts.kernels import Kernel
from gluonts.model.common import Tensor
from gluonts.support.linalg_util import (
    batch_diagonal,
    jitter_cholesky,
    jitter_cholesky_eig,
)


class GaussianProcess:
    # noinspection PyMethodOverriding, PyPep8Naming
    def __init__(
        self,
        sigma: Tensor,
        kernel: Kernel,
        prediction_length: Optional[int] = None,
        context_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        ctx: mx.Context = mx.Context("cpu"),
        float_type: DType = np.float64,
        jitter_method: str = "iter",
        max_iter_jitter: int = 10,
        neg_tol: float = -1e-8,
        diag_weight: float = 1e-6,
        increase_jitter: int = 10,
        sample_noise: bool = True,
        F=None,
    ) -> None:
        r"""
        Parameters
        ----------
        sigma
            Noise parameter of shape (batch_size, num_data_points, 1),
            where num_data_points is the number of rows in the Cholesky matrix.
        kernel
            Kernel object.
        prediction_length
            Prediction length.
        context_length
            Training length.
        num_samples
            The number of samples to be drawn.
        ctx
            Determines whether to compute on the cpu or gpu.
        float_type
            Determines whether to use single or double precision.
        jitter_method
            Iteratively jitter method or use eigenvalue decomposition depending on problem size.
        max_iter_jitter
            Maximum number of iterations for jitter to iteratively make the matrix positive definite.
        neg_tol
            Parameter in the jitter methods to eliminate eliminate matrices with diagonal elements smaller than this
            when checking if a matrix is positive definite.
        diag_weight
            Multiple of mean of diagonal entries to initialize the jitter.
        increase_jitter
            Each iteration multiply by jitter by this amount
        sample_noise
            Boolean to determine whether to add :math:`\sigma^2I` to the predictive covariance matrix.
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        """
        assert (
            prediction_length is None or prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert (
            num_samples is None or num_samples > 0
        ), "The value of `num_samples` should be > 0"
        self.sigma = sigma
        self.kernel = kernel
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.num_samples = num_samples
        self.F = F if F else getF(sigma)
        self.ctx = ctx
        self.float_type = float_type
        self.jitter_method = jitter_method
        self.max_iter_jitter = max_iter_jitter
        self.neg_tol = neg_tol
        self.diag_weight = diag_weight
        self.increase_jitter = increase_jitter
        self.sample_noise = sample_noise

    # noinspection PyMethodOverriding,PyPep8Naming
    def _compute_cholesky_gp(
        self,
        kernel_matrix: Tensor,
        num_data_points: Optional[int] = None,
        noise: bool = True,
    ) -> Tensor:
        r"""
        Parameters
        --------------------
        kernel_matrix
            Kernel matrix of shape (batch_size, num_data_points, num_data_points).
        num_data_points
            Number of rows in the kernel_matrix.
        noise
            Boolean to determine whether to add :math:`\sigma^2I` to the kernel matrix.
            This is used in the predictive step if you would like to sample the predictive
            covariance matrix without noise.  It is set to True in every other case.
        Returns
        --------------------
        Tensor
            Cholesky factor :math:`L` of the kernel matrix with added noise :math:`LL^T = K + \sigma^2 I`
            of shape (batch_size, num_data_points, num_data_points).
        """
        if noise:  # Add sigma
            kernel_matrix = self.F.broadcast_plus(
                kernel_matrix,
                self.F.broadcast_mul(
                    self.sigma ** 2,
                    self.F.eye(
                        num_data_points, ctx=self.ctx, dtype=self.float_type
                    ),
                ),
            )
        # Warning: This method is more expensive than the iterative jitter
        # but it works for mx.sym
        if self.jitter_method == "eig":
            return jitter_cholesky_eig(
                self.F,
                kernel_matrix,
                num_data_points,
                self.ctx,
                self.float_type,
                self.diag_weight,
            )
        elif self.jitter_method == "iter" and self.F is mx.nd:
            return jitter_cholesky(
                self.F,
                kernel_matrix,
                num_data_points,
                self.ctx,
                self.float_type,
                self.max_iter_jitter,
                self.neg_tol,
                self.diag_weight,
                self.increase_jitter,
            )
        else:
            return self.F.linalg.potrf(kernel_matrix)

    def log_prob(self, x_train: Tensor, y_train: Tensor) -> Tensor:
        r"""
        This method computes the negative marginal log likelihood
        
        .. math::
            :nowrap:

                \begin{aligned}
                    \frac{1}{2} [d \log(2\pi) + \log(|K|) + y^TK^{-1}y],
                \end{aligned}

        where :math:`d` is the number of data points.
        This can be written in terms of the Cholesky factor  :math:`L` as

        .. math::
            :nowrap:

            \begin{aligned}
                \log(|K|) = \log(|LL^T|) &= \log(|L||L|^T) = \log(|L|^2) = 2\log(|L|) \\
                &= 2\log\big(\prod_i^n L_{ii}\big) = 2 \sum_i^N \log(L_{ii})
            \end{aligned}
                 and

        .. math::
            :nowrap:

                 \begin{aligned}
                    y^TK^{-1}y = (y^TL^{-T})(L^{-1}y) = (L^{-1}y)^T(L^{-1}y) = ||L^{-1}y||_2^2.
                \end{aligned}

        Parameters
        --------------------
        x_train
            Training set of features of shape (batch_size, context_length, num_features).
        y_train
            Training labels of shape (batch_size, context_length).

        Returns
        --------------------
        Tensor
            The negative log marginal likelihood of shape (batch_size,)
        """
        assert (
            self.context_length is not None
        ), "The value of `context_length` must be set."
        return -MultivariateGaussian(
            self.F.zeros_like(y_train),  # 0 mean gaussian process prior
            self._compute_cholesky_gp(
                self.kernel.kernel_matrix(x_train, x_train),
                self.context_length,
            ),
        ).log_prob(y_train)

    def sample(self, mean: Tensor, covariance: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        covariance
            The covariance matrix of the GP of shape (batch_size, prediction_length, prediction_length).
        mean
            The mean vector of the GP of shape (batch_size, prediction_length).
        Returns
        -------
        Tensor
            Samples from a Gaussian Process of shape (batch_size, prediction_length, num_samples), where :math:`L`
            is the matrix square root, Cholesky Factor of the covariance matrix with the added noise tolerance on the
            diagonal, :math:`Lz`, where :math:`z \sim N(0,I)` and assumes the mean is zero.
        """
        assert (
            self.num_samples is not None
        ), "The value of `num_samples` must be set."
        assert (
            self.prediction_length is not None
        ), "The value of `prediction_length` must be set."
        samples = MultivariateGaussian(
            mean,
            self._compute_cholesky_gp(
                covariance, self.prediction_length, self.sample_noise
            ),
        ).sample_rep(
            self.num_samples, dtype=self.float_type
        )  # Shape (num_samples, batch_size, prediction_length)
        return self.F.transpose(samples, axes=(1, 2, 0))

    # noinspection PyMethodOverriding,PyPep8Naming
    def exact_inference(
        self, x_train: Tensor, y_train: Tensor, x_test: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x_train
            Training set of features of shape (batch_size, context_length, num_features).
        y_train
            Training labels of shape (batch_size, context_length).
        x_test
            Test set of features of shape (batch_size, prediction_length, num_features).
        Returns
        -------
        Tuple
            Tensor
                Predictive GP samples of shape (batch_size, prediction_length, num_samples).
            Tensor
                Predictive mean of the GP of shape (batch_size, prediction_length).
            Tensor
                Predictive standard deviation of the GP of shape (batch_size, prediction_length).
        """
        assert (
            self.context_length is not None
        ), "The value of `context_length` must be set."
        assert (
            self.prediction_length is not None
        ), "The value of `prediction_length` must be set."
        # Compute Cholesky factorization of training kernel matrix
        l_train = self._compute_cholesky_gp(
            self.kernel.kernel_matrix(x_train, x_train), self.context_length
        )

        lower_tri_solve = self.F.linalg.trsm(
            l_train, self.kernel.kernel_matrix(x_train, x_test)
        )
        predictive_mean = self.F.linalg.gemm2(
            lower_tri_solve,
            self.F.linalg.trsm(l_train, y_train.expand_dims(axis=-1)),
            transpose_a=True,
        ).squeeze(axis=-1)
        # Can rewrite second term as
        # :math:`||L^-1 * K(x_train,x_test||_2^2`
        #  and only solve 1 equation
        predictive_covariance = self.kernel.kernel_matrix(
            x_test, x_test
        ) - self.F.linalg.gemm2(
            lower_tri_solve, lower_tri_solve, transpose_a=True
        )
        # Extract diagonal entries of covariance matrix
        predictive_std = batch_diagonal(
            self.F,
            predictive_covariance,
            self.prediction_length,
            self.ctx,
            self.float_type,
        )
        # If self.sample_noise = True, predictive covariance has sigma^2 on the diagonal
        if self.sample_noise:
            predictive_std = self.F.broadcast_add(
                predictive_std, self.sigma ** 2
            )
        predictive_std = self.F.sqrt(predictive_std).squeeze(axis=-1)
        # Compute sample from GP predictive distribution
        return (
            self.sample(predictive_mean, predictive_covariance),
            predictive_mean,
            predictive_std,
        )

    @staticmethod
    def plot(
        ts_idx: int,
        x_train: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        x_test: Optional[Tensor] = None,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        samples: Optional[Tensor] = None,
        axis: Optional[List] = None,
    ) -> None:
        """
        This method plots the sampled GP distribution at the test points in solid colors, as well as the predictive
        mean as the dashed red line.  Plus and minus 2 predictive standard deviations are shown in the grey region.
        The training points are shown as the blue dots.

        Parameters
        ----------
        ts_idx
            Time series index to plot
        x_train
            Training set of features of shape (batch_size, context_length, num_features).
        y_train
            Training labels of shape (batch_size, context_length).
        x_test
            Test set of features of shape (batch_size, prediction_length, num_features).
        mean
             Mean of the GP of shape (batch_size, prediction_length).
        std
            Standard deviation of the GP of shape (batch_size, prediction_length, 1).
        samples
            GP samples of shape (batch_size, prediction_length, num_samples).
        axis
            Plot axes limits
        """

        # matplotlib==2.0.* gives errors in Brazil builds and has to be
        # imported locally
        import matplotlib.pyplot as plt

        if x_train is not None:
            x_train = x_train[ts_idx, :, :].asnumpy()
            if y_train is not None:
                y_train = y_train[ts_idx, :].asnumpy()
                plt.plot(x_train, y_train, "bs", ms=8)
        if x_test is not None:
            x_test = x_test[ts_idx, :, :].asnumpy()
            if samples is not None:
                samples = samples[ts_idx, :, :].asnumpy()
                plt.plot(x_test, samples)
            if mean is not None:
                mean = mean[ts_idx, :].asnumpy()
                plt.plot(x_test, mean, "r--", lw=2)
                if std is not None:
                    std = std[ts_idx, :].asnumpy()
                    plt.gca().fill_between(
                        x_test.flat,
                        mean - 2 * std,
                        mean + 2 * std,
                        color="#dddddd",
                    )
        if axis is not None:
            plt.axis(axis)
        plt.title(f"Samples from GP for time series {ts_idx}")
        plt.show()
