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

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions import NegativeBinomial as TorchNegativeBinomial

from gluonts.util import lazy_property
from scipy.stats import nbinom


from .distribution_output import DistributionOutput


class NegativeBinomial(TorchNegativeBinomial):
    """
    Negative binomial distribution with `total_count` and `probs` or `logits` parameters.

    Based on torch.distributions.NegativeBinomial, with added `cdf` and `icdf` methods.
    """

    def __init__(
        self,
        total_count: Union[float, torch.Tensor],
        probs: Optional[Union[float, torch.Tensor]] = None,
        logits: Optional[Union[float, torch.Tensor]] = None,
        validate_args=None,
    ):
        super().__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=validate_args,
        )

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        result = self.scipy_nbinom.cdf(value.detach().cpu().numpy())
        return torch.tensor(result, device=value.device, dtype=value.dtype)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        result = self.scipy_nbinom.ppf(value.detach().cpu().numpy())
        return torch.tensor(result, device=value.device, dtype=value.dtype)

    @lazy_property
    def scipy_nbinom(self):
        return nbinom(
            n=self.total_count.detach().cpu().numpy(),
            p=1.0 - self.probs.detach().cpu().numpy(),
        )


class NegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"total_count": 1, "logits": 1}
    distr_cls: type = NegativeBinomial

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
        total_count = F.softplus(total_count)
        return total_count.squeeze(-1), logits.squeeze(-1)

    def _base_distribution(self, distr_args) -> Distribution:
        total_count, logits = distr_args
        return self.distr_cls(total_count=total_count, logits=logits)

    # Overwrites the parent class method. We cannot scale using the affine
    # transformation since negative binomial should return integers. Instead
    # we scale the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        total_count, logits = distr_args

        if scale is not None:
            logits += scale.log()

        return NegativeBinomial(total_count=total_count, logits=logits)

    @property
    def event_shape(self) -> Tuple:
        return ()
