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

from typing import Dict, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .beta import Beta
from .deterministic import Deterministic
from .distribution import getF, softplus
from .distribution_output import DistributionOutput
from .mixture import MixtureDistribution


class ZeroAndOneInflatedBeta(MixtureDistribution):
    r"""
    Zero And One Inflated Beta distribution as in Raydonal Ospina, Silvia L.P. Ferrari: Inflated Beta Distributions

    Parameters
    ----------
    alpha
        Tensor containing the alpha shape parameters, of shape `(*batch_shape, *event_shape)`.
    beta
        Tensor containing the beta shape parameters, of shape `(*batch_shape, *event_shape)`.
    zero_probability
        Tensor containing the probability of zeros, of shape `(*batch_shape, *event_shape)`.
    one_probability
        Tensor containing the probability of ones, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(
        self,
        alpha: Tensor,
        beta: Tensor,
        zero_probability: Tensor,
        one_probability: Tensor,
    ) -> None:
        F = getF(alpha)
        self.alpha = alpha
        self.beta = beta
        self.zero_probability = zero_probability
        self.one_probability = one_probability
        self.beta_probability = 1 - zero_probability - one_probability
        self.beta_distribution = Beta(alpha=alpha, beta=beta)
        mixture_probs = F.stack(
            zero_probability, one_probability, self.beta_probability, axis=-1
        )
        super().__init__(
            components=[
                Deterministic(alpha.zeros_like()),
                Deterministic(alpha.ones_like()),
                self.beta_distribution,
            ],
            mixture_probs=mixture_probs,
        )

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F

        # mask zeros for the Beta distribution input to prevent NaN gradients
        inputs = F.where(
            F.broadcast_logical_or(x == 0, x == 1), x.zeros_like() + 0.5, x
        )

        # compute log density, case by case
        return F.where(
            x == 1,
            F.log(self.one_probability.broadcast_like(x)),
            F.where(
                x == 0,
                F.log(self.zero_probability.broadcast_like(x)),
                F.log(self.beta_probability)
                + self.beta_distribution.log_prob(inputs),
            ),
        )


class ZeroInflatedBeta(ZeroAndOneInflatedBeta):
    r"""
    Zero Inflated Beta distribution as in Raydonal Ospina, Silvia L.P. Ferrari: Inflated Beta Distributions

    Parameters
    ----------
    alpha
        Tensor containing the alpha shape parameters, of shape `(*batch_shape, *event_shape)`.
    beta
        Tensor containing the beta shape parameters, of shape `(*batch_shape, *event_shape)`.
    zero_probability
        Tensor containing the probability of zeros, of shape `(*batch_shape, *event_shape)`.
    F
    """
    is_reparameterizable = False

    @validated()
    def __init__(
        self, alpha: Tensor, beta: Tensor, zero_probability: Tensor
    ) -> None:
        super().__init__(
            alpha=alpha,
            beta=beta,
            zero_probability=zero_probability,
            one_probability=alpha.zeros_like(),
        )


class OneInflatedBeta(ZeroAndOneInflatedBeta):
    r"""
    One Inflated Beta distribution as in Raydonal Ospina, Silvia L.P. Ferrari: Inflated Beta Distributions

    Parameters
    ----------
    alpha
        Tensor containing the alpha shape parameters, of shape `(*batch_shape, *event_shape)`.
    beta
        Tensor containing the beta shape parameters, of shape `(*batch_shape, *event_shape)`.
    one_probability
        Tensor containing the probability of ones, of shape `(*batch_shape, *event_shape)`.
    F
    """
    is_reparameterizable = False

    @validated()
    def __init__(
        self, alpha: Tensor, beta: Tensor, one_probability: Tensor
    ) -> None:
        super().__init__(
            alpha=alpha,
            beta=beta,
            zero_probability=alpha.zeros_like(),
            one_probability=one_probability,
        )


class ZeroAndOneInflatedBetaOutput(DistributionOutput):
    args_dim: Dict[str, int] = {
        "alpha": 1,
        "beta": 1,
        "zero_probability": 1,
        "one_probability": 1,
    }
    distr_cls: type = ZeroAndOneInflatedBeta

    @classmethod
    def domain_map(cls, F, alpha, beta, zero_probability, one_probability):
        r"""
        Maps raw tensors to valid arguments for constructing a ZeroAndOneInflatedBeta
        distribution.

        Parameters
        ----------
        F:
        alpha:
            Tensor of shape `(*batch_shape, 1)`
        beta:
            Tensor of shape `(*batch_shape, 1)`
        zero_probability:
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            Four squeezed tensors, of shape `(*batch_shape)`: First two have entries mapped to the
            positive orthant, zero_probability is mapped to (0, 1), one_probability is mapped to (0, 1-zero_probability)
        """
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon

        alpha = softplus(F, alpha) + epsilon
        beta = softplus(F, beta) + epsilon
        zero_probability = F.sigmoid(zero_probability)
        one_probability = (1 - zero_probability) * F.sigmoid(one_probability)
        return (
            alpha.squeeze(axis=-1),
            beta.squeeze(axis=-1),
            zero_probability.squeeze(axis=-1),
            one_probability.squeeze(axis=-1),
        )

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def value_in_support(self) -> float:
        return 0.5


class ZeroInflatedBetaOutput(ZeroAndOneInflatedBetaOutput):
    args_dim: Dict[str, int] = {"alpha": 1, "beta": 1, "zero_probability": 1}
    distr_cls: type = ZeroInflatedBeta

    @classmethod
    def domain_map(cls, F, alpha, beta, zero_probability):
        r"""
        Maps raw tensors to valid arguments for constructing a ZeroInflatedBeta
        distribution.

        Parameters
        ----------
        F:
        alpha:
            Tensor of shape `(*batch_shape, 1)`
        beta:
            Tensor of shape `(*batch_shape, 1)`
        zero_probability:
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]:
            Three squeezed tensors, of shape `(*batch_shape)`: First two have entries mapped to the
            positive orthant, last is mapped to (0,1)
        """
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon

        alpha = softplus(F, alpha) + epsilon
        beta = softplus(F, beta) + epsilon
        zero_probability = F.sigmoid(zero_probability)
        return (
            alpha.squeeze(axis=-1),
            beta.squeeze(axis=-1),
            zero_probability.squeeze(axis=-1),
        )


class OneInflatedBetaOutput(ZeroInflatedBetaOutput):
    args_dim: Dict[str, int] = {"alpha": 1, "beta": 1, "one_probability": 1}
    distr_cls: type = OneInflatedBeta

    @classmethod
    def domain_map(cls, F, alpha, beta, one_probability):
        return super().domain_map(F, alpha, beta, one_probability)
