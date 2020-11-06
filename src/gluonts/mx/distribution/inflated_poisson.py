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
from typing import Dict, Tuple

# Third-party imports
import numpy as np
from gluonts.core.component import validated

# First-party imports
from gluonts.model.common import Tensor

# Relative imports
from .distribution import getF, softplus
from .distribution_output import DistributionOutput
from .poisson import Poisson, PoissonOutput
from .mixture import MixtureDistribution, MixtureDistributionOutput
from .deterministic import Deterministic, DeterministicOutput

class ZeroInflatedPoisson(MixtureDistribution):
    r"""
    Zero Inflated Poisson distribution TODO add reference
    Parameters
    ----------
    rate
        Tensor containing the rate, of shape `(*batch_shape, *event_shape)`.
    zero_probability
        Tensor containing the probability of zeros, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, rate: Tensor, zero_probability: Tensor) -> None:

        F = getF(rate)
        self.rate = rate
        self.zero_probability = zero_probability
        self.poisson_distribution = Poisson(rate=rate)
        mixture_probs = F.stack(
            zero_probability, 1 - zero_probability, axis=-1
        )
        super().__init__(
            components=[
                Deterministic(rate.zeros_like()),
                self.poisson_distribution,
            ],
            mixture_probs=mixture_probs,
        )

    def log_prob(self, x):
        F = self.F

        # log_prob of zeros
        zero_likelihood = self.zero_probability + (
            1 - self.zero_probability
        ) * F.exp(-self.rate)

        return F.where(
            x == 0,
            F.log(zero_likelihood.broadcast_like(x)),
            F.log(1 - self.zero_probability) + self.poisson_distribution.log_prob(x),
        )


# class ZeroInflatedPoissonOutput(MixtureDistributionOutput):
#     def __init__(self):
#         super().__init__([DeterministicOutput(0), PoissonOutput()])


class ZeroInflatedPoissonOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1, "zero_probability": 1}
    distr_cls: type = ZeroInflatedPoisson

    @classmethod
    def domain_map(cls, F, rate, zero_probability):

        epsilon = np.finfo(cls._dtype).eps  # machine epsilon

        rate = softplus(F, rate) + epsilon
        zero_probability = F.sigmoid(zero_probability)
        return rate.squeeze(axis=-1), zero_probability.squeeze(axis=-1)
