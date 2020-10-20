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
from typing import Optional, Tuple

# Third-party imports
import mxnet as mx
import numpy as np

from gluonts.core.component import validated

# First-party imports
from gluonts.model.common import Tensor
from gluonts.support.linalg_util import jitter_cholesky
from gluonts.support.util import _broadcast_param, make_nd_diag

from . import Distribution, Gaussian, MultivariateGaussian
from .distribution import getF


class ParameterBounds:
    @validated()
    def __init__(self, lower, upper) -> None:
        assert (
            lower <= upper
        ), "lower bound should be smaller or equal to upper bound"
        self.lower = lower
        self.upper = upper


def _safe_split(x, num_outputs, axis, squeeze_axis, *args, **kwargs):
    """
    A type-stable wrapper around mx.nd.split.

    Currently mx.nd.split behaves weirdly if num_outputs=1:

        a = mx.nd.ones(shape=(1, 1, 2))
        l = a.split(axis=1, num_outputs=1, squeeze_axis=False)
        type(l)  # mx.NDArray
        l.shape  # (1, 1, 2)

    Compare that with the case num_outputs=2:

        a = mx.nd.ones(shape=(1, 2, 2))
        l = a.split(axis=1, num_outputs=2, squeeze_axis=False)
        type(l)  # list
        len(l)  # 2
        l[0].shape  # (1, 1, 2)

    This wrapper makes the behavior consistent by always returning a list
    of length num_outputs, whose elements will have one less axis than x
    in case x.shape[axis]==num_outputs and squeeze_axis==True, and the same
    number of axes as x otherwise.
    """
    if num_outputs > 1:
        return x.split(
            axis=axis,
            num_outputs=num_outputs,
            squeeze_axis=squeeze_axis,
            *args,
            **kwargs
        )
    return [x.squeeze(axis=axis)] if squeeze_axis else [x]


class LDS(Distribution):
    r"""
    Implements Linear Dynamical System (LDS) as a distribution.

    The LDS is given by

    .. math::
        z_t = A_t l_{t-1} + b_t + \epsilon_t \\
        l_t = C_t l_{t-1} + g_t \nu

    where

    .. math::
        \epsilon_t = N(0, S_v) \\
        \nu = N(0, 1)

    :math:`A_t`, :math:`C_t` and :math:`g_t` are the emission, transition and
    innovation coefficients respectively. The residual terms are denoted
    by :math:`b_t`.

    The target :math:`z_t` can be :math:`d`-dimensional in which case

    .. math::
        A_t \in R^{d \times h}, b_t \in R^{d}, C_t \in R^{h \times h}, g_t \in R^{h}

    where :math:`h` is dimension of the latent state.

    Parameters
    ----------
    emission_coeff
        Tensor of shape (batch_size, seq_length, obs_dim, latent_dim)
    transition_coeff
        Tensor of shape (batch_size, seq_length, latent_dim, latent_dim)
    innovation_coeff
        Tensor of shape (batch_size, seq_length, latent_dim)
    noise_std
        Tensor of shape (batch_size, seq_length, obs_dim)
    residuals
        Tensor of shape (batch_size, seq_length, obs_dim)
    prior_mean
        Tensor of shape (batch_size, latent_dim)
    prior_cov
        Tensor of shape (batch_size, latent_dim, latent_dim)
    latent_dim
        Dimension of the latent state
    output_dim
        Dimension of the output
    seq_length
        Sequence length
    F
    """

    @validated()
    def __init__(
        self,
        emission_coeff: Tensor,
        transition_coeff: Tensor,
        innovation_coeff: Tensor,
        noise_std: Tensor,
        residuals: Tensor,
        prior_mean: Tensor,
        prior_cov: Tensor,
        latent_dim: int,
        output_dim: int,
        seq_length: int,
    ) -> None:
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_length = seq_length

        # Split coefficients along time axis for easy access
        # emission_coef[t]: (batch_size, obs_dim, latent_dim)
        self.emission_coeff = _safe_split(
            emission_coeff,
            axis=1,
            num_outputs=self.seq_length,
            squeeze_axis=True,
        )

        # innovation_coef[t]: (batch_size, latent_dim)
        self.innovation_coeff = _safe_split(
            innovation_coeff,
            axis=1,
            num_outputs=self.seq_length,
            squeeze_axis=False,
        )

        # transition_coeff: (batch_size, latent_dim, latent_dim)
        self.transition_coeff = _safe_split(
            transition_coeff,
            axis=1,
            num_outputs=self.seq_length,
            squeeze_axis=True,
        )

        # noise_std[t]: (batch_size, obs_dim)
        self.noise_std = _safe_split(
            noise_std, axis=1, num_outputs=self.seq_length, squeeze_axis=True
        )

        # residuals[t]: (batch_size, obs_dim)
        self.residuals = _safe_split(
            residuals, axis=1, num_outputs=self.seq_length, squeeze_axis=True
        )

        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

    @property
    def F(self):
        return getF(self.prior_mean)

    @property
    def batch_shape(self) -> Tuple:
        return self.emission_coeff[0].shape[:1] + (self.seq_length,)

    @property
    def event_shape(self) -> Tuple:
        return (self.output_dim,)

    @property
    def event_dim(self) -> int:
        return 2

    def log_prob(
        self,
        x: Tensor,
        scale: Optional[Tensor] = None,
        observed: Optional[Tensor] = None,
    ):
        """
        Compute the log probability of observations.

        This method also returns the final state of the system.

        Parameters
        ----------
        x
            Observations, shape (batch_size, seq_length, output_dim)
        scale
            Scale of each sequence in x, shape (batch_size, output_dim)
        observed
            Flag tensor indicating which observations are genuine (1.0) and
            which are missing (0.0)

        Returns
        -------
        Tensor
            Log probabilities, shape (batch_size, seq_length)
        Tensor
            Final mean, shape (batch_size, latent_dim)
        Tensor
            Final covariance, shape (batch_size, latent_dim, latent_dim)
        """
        if scale is not None:
            x = self.F.broadcast_div(x, scale.expand_dims(axis=1))
        # TODO: Based on form of the prior decide to do either filtering
        #   or residual-sum-of-squares
        log_p, final_mean, final_cov = self.kalman_filter(x, observed)
        return log_p, final_mean, final_cov

    def kalman_filter(
        self, targets: Tensor, observed: Tensor
    ) -> Tuple[Tensor, ...]:
        """
        Performs Kalman filtering given observations.


        Parameters
        ----------
        targets
            Observations, shape (batch_size, seq_length, output_dim)
        observed
            Flag tensor indicating which observations are genuine (1.0) and
            which are missing (0.0)

        Returns
        -------
        Tensor
            Log probabilities, shape (batch_size, seq_length)
        Tensor
            Mean of p(l_T | l_{T-1}), where T is seq_length, with shape
            (batch_size, latent_dim)
        Tensor
            Covariance of p(l_T | l_{T-1}), where T is seq_length, with shape
            (batch_size, latent_dim, latent_dim)
        """
        F = self.F
        # targets[t]: (batch_size, obs_dim)
        targets = _safe_split(
            targets, axis=1, num_outputs=self.seq_length, squeeze_axis=True
        )

        log_p_seq = []

        mean = self.prior_mean
        cov = self.prior_cov

        observed = (
            _safe_split(
                observed,
                axis=1,
                num_outputs=self.seq_length,
                squeeze_axis=True,
            )
            if observed is not None
            else None
        )

        for t in range(self.seq_length):
            # Compute the filtered distribution
            #   p(l_t | z_1, ..., z_{t + 1})
            # and log - probability
            #   log p(z_t | z_0, z_{t - 1})
            filtered_mean, filtered_cov, log_p = kalman_filter_step(
                F,
                target=targets[t],
                prior_mean=mean,
                prior_cov=cov,
                emission_coeff=self.emission_coeff[t],
                residual=self.residuals[t],
                noise_std=self.noise_std[t],
                latent_dim=self.latent_dim,
                output_dim=self.output_dim,
            )

            log_p_seq.append(log_p.expand_dims(axis=1))

            # Mean of p(l_{t+1} | l_t)
            mean = F.linalg_gemm2(
                self.transition_coeff[t],
                (
                    filtered_mean.expand_dims(axis=-1)
                    if observed is None
                    else F.where(
                        observed[t], x=filtered_mean, y=mean
                    ).expand_dims(axis=-1)
                ),
            ).squeeze(axis=-1)

            # Covariance of p(l_{t+1} | l_t)
            cov = F.linalg_gemm2(
                self.transition_coeff[t],
                F.linalg_gemm2(
                    (
                        filtered_cov
                        if observed is None
                        else F.where(observed[t], x=filtered_cov, y=cov)
                    ),
                    self.transition_coeff[t],
                    transpose_b=True,
                ),
            ) + F.linalg_gemm2(
                self.innovation_coeff[t],
                self.innovation_coeff[t],
                transpose_a=True,
            )

        # Return sequence of log likelihoods, as well as
        # final mean and covariance of p(l_T | l_{T-1} where T is seq_length
        return F.concat(*log_p_seq, dim=1), mean, cov

    def sample(
        self, num_samples: Optional[int] = None, scale: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Generates samples from the LDS: p(z_1, z_2, \ldots, z_{`seq_length`}).

        Parameters
        ----------
        num_samples
            Number of samples to generate
        scale
            Scale of each sequence in x, shape (batch_size, output_dim)

        Returns
        -------
        Tensor
            Samples, shape (num_samples, batch_size, seq_length, output_dim)
        """
        F = self.F

        # Note on shapes: here we work with tensors of the following shape
        # in each time step t: (num_samples, batch_size, dim, dim),
        # where dim can be obs_dim or latent_dim or a constant 1 to facilitate
        # generalized matrix multiplication (gemm2)

        # Sample observation noise for all time steps
        # noise_std: (batch_size, seq_length, obs_dim, 1)
        noise_std = F.stack(*self.noise_std, axis=1).expand_dims(axis=-1)

        # samples_eps_obs[t]: (num_samples, batch_size, obs_dim, 1)
        samples_eps_obs = _safe_split(
            Gaussian(noise_std.zeros_like(), noise_std).sample(num_samples),
            axis=-3,
            num_outputs=self.seq_length,
            squeeze_axis=True,
        )

        # Sample standard normal for all time steps
        # samples_eps_std_normal[t]: (num_samples, batch_size, obs_dim, 1)
        samples_std_normal = _safe_split(
            Gaussian(noise_std.zeros_like(), noise_std.ones_like()).sample(
                num_samples
            ),
            axis=-3,
            num_outputs=self.seq_length,
            squeeze_axis=True,
        )

        # Sample the prior state.
        # samples_lat_state: (num_samples, batch_size, latent_dim, 1)
        # The prior covariance is observed to be slightly negative definite whenever there is
        # excessive zero padding at the beginning of the time series.
        # We add positive tolerance to the diagonal to avoid numerical issues.
        # Note that `jitter_cholesky` adds positive tolerance only if the decomposition without jitter fails.
        state = MultivariateGaussian(
            self.prior_mean,
            jitter_cholesky(
                F, self.prior_cov, self.latent_dim, float_type=np.float32
            ),
        )
        samples_lat_state = state.sample(num_samples).expand_dims(axis=-1)

        samples_seq = []
        for t in range(self.seq_length):
            # Expand all coefficients to include samples in axis 0
            # emission_coeff_t: (num_samples, batch_size, obs_dim, latent_dim)
            # transition_coeff_t:
            #   (num_samples, batch_size, latent_dim, latent_dim)
            # innovation_coeff_t: (num_samples, batch_size, 1, latent_dim)
            emission_coeff_t, transition_coeff_t, innovation_coeff_t = [
                _broadcast_param(coeff, axes=[0], sizes=[num_samples])
                if num_samples is not None
                else coeff
                for coeff in [
                    self.emission_coeff[t],
                    self.transition_coeff[t],
                    self.innovation_coeff[t],
                ]
            ]

            # Expand residuals as well
            # residual_t: (num_samples, batch_size, obs_dim, 1)
            residual_t = (
                _broadcast_param(
                    self.residuals[t].expand_dims(axis=-1),
                    axes=[0],
                    sizes=[num_samples],
                )
                if num_samples is not None
                else self.residuals[t].expand_dims(axis=-1)
            )

            # (num_samples, batch_size, 1, obs_dim)
            samples_t = (
                F.linalg_gemm2(emission_coeff_t, samples_lat_state)
                + residual_t
                + samples_eps_obs[t]
            )
            samples_t = (
                samples_t.swapaxes(dim1=2, dim2=3)
                if num_samples is not None
                else samples_t.swapaxes(dim1=1, dim2=2)
            )
            samples_seq.append(samples_t)

            # sample next state: (num_samples, batch_size, latent_dim, 1)
            samples_lat_state = F.linalg_gemm2(
                transition_coeff_t, samples_lat_state
            ) + F.linalg_gemm2(
                innovation_coeff_t, samples_std_normal[t], transpose_a=True
            )

        # (num_samples, batch_size, seq_length, obs_dim)
        samples = F.concat(*samples_seq, dim=-2)
        return (
            samples
            if scale is None
            else F.broadcast_mul(
                samples,
                scale.expand_dims(axis=1).expand_dims(axis=0)
                if num_samples is not None
                else scale.expand_dims(axis=1),
            )
        )

    def sample_marginals(
        self, num_samples: Optional[int] = None, scale: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Generates samples from the marginals p(z_t),
        t = 1, \ldots, `seq_length`.

        Parameters
        ----------
        num_samples
            Number of samples to generate
        scale
            Scale of each sequence in x, shape (batch_size, output_dim)

        Returns
        -------
        Tensor
            Samples, shape (num_samples, batch_size, seq_length, output_dim)
        """
        F = self.F

        state_mean = self.prior_mean.expand_dims(axis=-1)
        state_cov = self.prior_cov

        output_mean_seq = []
        output_cov_seq = []

        for t in range(self.seq_length):
            # compute and store observation mean at time t
            output_mean = F.linalg_gemm2(
                self.emission_coeff[t], state_mean
            ) + self.residuals[t].expand_dims(axis=-1)

            output_mean_seq.append(output_mean)

            # compute and store observation cov at time t
            output_cov = F.linalg_gemm2(
                self.emission_coeff[t],
                F.linalg_gemm2(
                    state_cov, self.emission_coeff[t], transpose_b=True
                ),
            ) + make_nd_diag(
                F=F, x=self.noise_std[t] * self.noise_std[t], d=self.output_dim
            )

            output_cov_seq.append(output_cov.expand_dims(axis=1))

            state_mean = F.linalg_gemm2(self.transition_coeff[t], state_mean)

            state_cov = F.linalg_gemm2(
                self.transition_coeff[t],
                F.linalg_gemm2(
                    state_cov, self.transition_coeff[t], transpose_b=True
                ),
            ) + F.linalg_gemm2(
                self.innovation_coeff[t],
                self.innovation_coeff[t],
                transpose_a=True,
            )

        output_mean = F.concat(*output_mean_seq, dim=1)
        output_cov = F.concat(*output_cov_seq, dim=1)

        L = F.linalg_potrf(output_cov)

        output_distribution = MultivariateGaussian(output_mean, L)

        samples = output_distribution.sample(num_samples=num_samples)

        return (
            samples
            if scale is None
            else F.broadcast_mul(samples, scale.expand_dims(axis=1))
        )


class LDSArgsProj(mx.gluon.HybridBlock):
    def __init__(
        self,
        output_dim: int,
        noise_std_bounds: ParameterBounds,
        innovation_bounds: ParameterBounds,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.dense_noise_std = mx.gluon.nn.Dense(
            units=1, flatten=False, activation="sigmoid"
        )
        self.dense_innovation = mx.gluon.nn.Dense(
            units=1, flatten=False, activation="sigmoid"
        )
        self.dense_residual = mx.gluon.nn.Dense(
            units=output_dim, flatten=False
        )

        self.innovation_bounds = innovation_bounds
        self.noise_std_bounds = noise_std_bounds

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        noise_std = (
            self.dense_noise_std(x)
            * (self.noise_std_bounds.upper - self.noise_std_bounds.lower)
            + self.noise_std_bounds.lower
        )

        innovation = (
            self.dense_innovation(x)
            * (self.innovation_bounds.upper - self.innovation_bounds.lower)
            + self.innovation_bounds.lower
        )

        residual = self.dense_residual(x)

        return noise_std, innovation, residual


def kalman_filter_step(
    F,
    target: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    emission_coeff: Tensor,
    residual: Tensor,
    noise_std: Tensor,
    latent_dim: int,
    output_dim: int,
):
    """
    One step of the Kalman filter.

    This function computes the filtered state (mean and covariance) given the
    linear system coefficients the prior state (mean and variance),
    as well as observations.

    Parameters
    ----------
    F
    target
        Observations of the system output, shape (batch_size, output_dim)
    prior_mean
        Prior mean of the latent state, shape (batch_size, latent_dim)
    prior_cov
        Prior covariance of the latent state, shape
        (batch_size, latent_dim, latent_dim)
    emission_coeff
        Emission coefficient, shape (batch_size, output_dim, latent_dim)
    residual
        Residual component, shape (batch_size, output_dim)
    noise_std
        Standard deviation of the output noise, shape (batch_size, output_dim)
    latent_dim
        Dimension of the latent state vector
    Returns
    -------
    Tensor
        Filtered_mean, shape (batch_size, latent_dim)
    Tensor
        Filtered_covariance, shape (batch_size, latent_dim, latent_dim)
    Tensor
        Log probability, shape (batch_size, )
    """
    # output_mean: mean of the target (batch_size, obs_dim)
    output_mean = F.linalg_gemm2(
        emission_coeff, prior_mean.expand_dims(axis=-1)
    ).squeeze(axis=-1)

    # noise covariance
    noise_cov = make_nd_diag(F=F, x=noise_std * noise_std, d=output_dim)

    S_hh_x_A_tr = F.linalg_gemm2(prior_cov, emission_coeff, transpose_b=True)

    # covariance of the target
    output_cov = F.linalg_gemm2(emission_coeff, S_hh_x_A_tr) + noise_cov

    # compute the Cholesky decomposition output_cov = LL^T
    L_output_cov = F.linalg_potrf(output_cov)

    # Compute Kalman gain matrix K:
    # K = S_hh X with X = A^T output_cov^{-1}
    # We have X = A^T output_cov^{-1} => X output_cov = A^T => X LL^T = A^T
    # We can thus obtain X by solving two linear systems involving L
    kalman_gain = F.linalg_trsm(
        L_output_cov,
        F.linalg_trsm(
            L_output_cov, S_hh_x_A_tr, rightside=True, transpose=True
        ),
        rightside=True,
    )

    # compute the error
    target_minus_residual = target - residual
    delta = target_minus_residual - output_mean

    # filtered estimates
    filtered_mean = prior_mean.expand_dims(axis=-1) + F.linalg_gemm2(
        kalman_gain, delta.expand_dims(axis=-1)
    )
    filtered_mean = filtered_mean.squeeze(axis=-1)

    # Joseph's symmetrized update for covariance:
    ImKA = F.broadcast_sub(
        F.eye(latent_dim), F.linalg_gemm2(kalman_gain, emission_coeff)
    )

    filtered_cov = F.linalg_gemm2(
        ImKA, F.linalg_gemm2(prior_cov, ImKA, transpose_b=True)
    ) + F.linalg_gemm2(
        kalman_gain, F.linalg_gemm2(noise_cov, kalman_gain, transpose_b=True)
    )

    # likelihood term: (batch_size,)
    log_p = MultivariateGaussian(output_mean, L_output_cov).log_prob(
        target_minus_residual
    )

    return filtered_mean, filtered_cov, log_p
