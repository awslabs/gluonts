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

"""
Tweedie distribution implementation for GluonTS.

The Tweedie distribution is a family of probability distributions that includes
Gaussian, Poisson, Gamma, and compound Poisson-Gamma as special cases. It is
particularly useful for modeling non-negative data with exact zeros, such as
insurance claims, rainfall amounts, and intermittent demand.

The distribution is parameterized by:
- mu: mean (must be positive)
- dispersion: dispersion parameter (must be positive)
- power: variance power parameter (must be in range (1, 2) for compound Poisson-Gamma)

The variance is: Var(Y) = dispersion * mu^power

This implementation is based on:
- PyTorch PR #171705: https://github.com/pytorch/pytorch/pull/171705
- Dunn, P.K. and Smyth, G.K. (2005). "Series evaluation of Tweedie exponential
  dispersion model densities". Statistics and Computing, 15(4), 267-280.
- R statmod package for the saddle-point approximation approach.
"""

from typing import Dict, Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, Poisson, constraints

from .distribution_output import DistributionOutput


class Tweedie(Distribution):
    """
    Tweedie distribution with compound Poisson-Gamma representation.

    The Tweedie distribution is characterized by the variance function:
        Var(Y) = dispersion * mu^power

    For power in (1, 2), this is a compound Poisson-Gamma distribution with
    a point mass at zero.

    Parameters
    ----------
    mu
        Mean parameter (must be positive).
    dispersion
        Dispersion parameter (must be positive).
    power
        Variance power parameter (must be in (1, 2)).
    validate_args
        Whether to validate input arguments.
    """

    arg_constraints = {
        "mu": constraints.positive,
        "dispersion": constraints.positive,
        "power": constraints.interval(1.0, 2.0),
    }
    support = constraints.greater_than_eq(0)
    has_rsample = False

    def __init__(
        self,
        mu: torch.Tensor,
        dispersion: torch.Tensor,
        power: torch.Tensor,
        validate_args: Optional[bool] = None,
    ):
        self.mu = mu
        self.dispersion = dispersion
        self.power = power

        batch_shape = torch.broadcast_shapes(
            mu.shape, dispersion.shape, power.shape
        )
        event_shape = torch.Size([])
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        return self.dispersion * torch.pow(self.mu, self.power)

    def _lambda(self) -> torch.Tensor:
        """Poisson rate parameter for compound Poisson-Gamma representation."""
        return torch.pow(self.mu, 2 - self.power) / (
            self.dispersion * (2 - self.power)
        )

    def _alpha(self) -> torch.Tensor:
        """Gamma shape parameter for compound Poisson-Gamma representation."""
        return (2 - self.power) / (self.power - 1)

    def _beta(self) -> torch.Tensor:
        """Gamma rate parameter for compound Poisson-Gamma representation."""
        return 1.0 / (
            self.dispersion
            * (self.power - 1)
            * torch.pow(self.mu, self.power - 1)
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample from the Tweedie distribution using compound Poisson-Gamma.

        Algorithm:
        1. Sample N ~ Poisson(lambda)
        2. If N > 0: Sample Y = sum of N Gamma(alpha, beta) random variables
        3. If N = 0: Y = 0
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)

            lam = self._lambda().expand(shape)
            alpha = self._alpha().expand(shape)
            beta = self._beta().expand(shape)

            # Sample Poisson counts
            n_samples = torch.poisson(lam)

            # Initialize output
            y = torch.zeros_like(n_samples)

            # For non-zero counts, sample from Gamma
            # Use the property that sum of N Gamma(alpha, beta) = Gamma(N*alpha, beta)
            mask = n_samples > 0
            if mask.any():
                # Create Gamma distribution for non-zero samples
                # Gamma(N*alpha, beta) where we parameterize by shape and rate
                gamma_shape = n_samples[mask] * alpha[mask]
                gamma_rate = beta[mask]

                # PyTorch Gamma uses concentration (shape) and rate
                gamma_dist = Gamma(concentration=gamma_shape, rate=gamma_rate)
                y[mask] = gamma_dist.sample()

            return y

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of observed values.

        Uses the saddle-point approximation for numerical stability.
        Based on Dunn & Smyth (2005) and the implementation approach from
        the R statmod package.
        """
        if self._validate_args:
            self._validate_sample(value)

        # Broadcast parameters to match value shape
        mu = self.mu.expand(value.shape)
        phi = self.dispersion.expand(value.shape)
        p = self.power.expand(value.shape)

        # Initialize log_prob tensor
        log_prob = torch.zeros_like(value)

        # Handle zero values: P(Y=0) = exp(-lambda)
        # where lambda = mu^(2-p) / (phi * (2-p))
        zero_mask = value == 0
        if zero_mask.any():
            lam = torch.pow(mu, 2 - p) / (phi * (2 - p))
            log_prob = torch.where(zero_mask, -lam, log_prob)

        # Handle positive values using saddle-point approximation
        pos_mask = value > 0
        if pos_mask.any():
            # Compute the unit deviance
            # d(y, mu) = 2 * (y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
            y = value

            # For numerical stability, compute each term carefully
            term1 = torch.pow(y.clamp(min=1e-10), 2 - p) / ((1 - p) * (2 - p))
            term2 = y * torch.pow(mu, 1 - p) / (1 - p)
            term3 = torch.pow(mu, 2 - p) / (2 - p)

            deviance = 2 * (term1 - term2 + term3)

            # Log-likelihood using saddle-point approximation:
            # log f(y; mu, phi, p) = -deviance/(2*phi) - log(y) - 0.5*log(2*pi*phi*V(y))
            # where V(y) = y^p is the variance function evaluated at y

            # Variance function at y
            V_y = torch.pow(y.clamp(min=1e-10), p)

            # Saddle-point log-likelihood
            log_prob_pos = (
                -deviance / (2 * phi)
                - torch.log(y.clamp(min=1e-10))
                - 0.5 * torch.log(2 * math.pi * phi * V_y)
            )

            log_prob = torch.where(pos_mask, log_prob_pos, log_prob)

        return log_prob


class TweedieOutput(DistributionOutput):
    """
    Distribution output class for Tweedie distribution.

    Maps neural network outputs to Tweedie distribution parameters.
    """

    args_dim: Dict[str, int] = {"mu": 1, "dispersion": 1, "power": 1}
    distr_cls: type = Tweedie

    @classmethod
    def domain_map(
        cls, mu: torch.Tensor, dispersion: torch.Tensor, power: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map unbounded neural network outputs to valid distribution parameters.

        Parameters
        ----------
        mu
            Unbounded tensor, mapped to positive via softplus + epsilon.
        dispersion
            Unbounded tensor, mapped to positive via softplus + epsilon.
        power
            Unbounded tensor, mapped to (1, 2) via 1 + sigmoid.

        Returns
        -------
        Tuple of transformed tensors (mu, dispersion, power).
        """
        epsilon = torch.finfo(mu.dtype).eps

        # mu must be positive
        mu = F.softplus(mu) + epsilon

        # dispersion must be positive
        dispersion = F.softplus(dispersion) + epsilon

        # power must be in (1, 2)
        # Using 1 + sigmoid maps to approximately (1, 2)
        # Add small margin to avoid boundary issues
        power = 1.0 + 0.01 + 0.98 * torch.sigmoid(power)

        return mu.squeeze(-1), dispersion.squeeze(-1), power.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def value_in_support(self) -> float:
        return 0.5
