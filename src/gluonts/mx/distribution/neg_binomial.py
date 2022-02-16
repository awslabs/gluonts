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

from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .deterministic import DeterministicOutput
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput
from .mixture import MixtureDistributionOutput


class NegativeBinomial(Distribution):
    r"""
    Negative binomial distribution, i.e. the distribution of the number of
    successes in a sequence of independent Bernoulli trials.

    Parameters
    ----------
    count
        Non-negative Tensor containing the number of failed Bernoulli trials
        for the process to stop. Shape is `(*batch_shape, *event_shape)`.
    logit
        Tensor containing the log-odds of success in the Bernoulli trials.
        Shape is `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, count: Tensor, logit: Tensor) -> None:
        self.count = count
        self.logit = logit

    @property
    def F(self):
        return getF(self.count)

    @property
    def batch_shape(self) -> Tuple:
        return self.count.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F

        log_unnormalized_prob = (
            self.count * F.log(F.sigmoid(-self.logit)) +
            x * F.log(F.sigmoid(self.logit))
        )

        log_normalization = (
            F.gammaln(1. + x) +
            F.gammaln(self.count) -
            F.gammaln(self.count + x)
        )

        return log_unnormalized_prob - log_normalization

    @property
    def mean(self) -> Tensor:
        return self.count * self.F.exp(self.logit)

    @property
    def stddev(self) -> Tensor:
        return self.F.sqrt(self.mean / self.F.sigmoid(-self.logit))

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(count: Tensor, logit: Tensor) -> Tensor:
            F = self.F
            return F.random.poisson(
                lam=F.random.gamma(
                    count,
                    F.exp(logit),
                ),
                dtype=dtype
            )

        return _sample_multiple(
            s, count=self.count, logit=self.logit, num_samples=num_samples
        )

    @property
    def args(self) -> List:
        return [self.count, self.logit]


class NegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"count": 1, "logit": 1}
    distr_cls: type = NegativeBinomial

    @classmethod
    def domain_map(cls, F, count, logit):
        count = F.maximum(softplus(F, count), np.finfo(cls._dtype).eps)
        return count.squeeze(-1), logit.squeeze(-1)

    # Overwrites the parent class method.
    # We cannot scale using the affine transformation since negative binomial should return integers.
    # Instead we scale the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> NegativeBinomial:
        count, logit = distr_args

        if scale is not None:
            logit = getF(count).broadcast_add(logit, scale.log())

        return NegativeBinomial(count=count, logit=logit)

    @property
    def event_shape(self) -> Tuple:
        return ()


def ZeroInflatedNegativeBinomialOutput() -> MixtureDistributionOutput:
    return MixtureDistributionOutput(
        distr_outputs=[NegativeBinomialOutput(), DeterministicOutput(0)]
    )
