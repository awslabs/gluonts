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

import torch
from pydantic import BaseModel


class DistributionLoss(BaseModel):
    """
    A ``torch.nn.Module`` extensions that computes loss values by comparing a
    ``Distribution`` (prediction) to a ``Tensor`` (ground-truth).
    """

    def __call__(
        self, input: torch.distributions.Distribution, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss of predicting ``target`` with the ``input``
        distribution.

        Parameters
        ----------
        input
            Distribution object representing the prediction.
        target
            Tensor containing the ground truth.

        Returns
        -------
        torch.Tensor
            Tensor containing loss values, with the same shape as ``target``.

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError


class NegativeLogLikelihood(DistributionLoss):
    """
    Compute the negative log likelihood loss.

    Parameters
    ----------
    beta: float in range (0, 1)
        beta parameter from the paper: "On the Pitfalls of Heteroscedastic
        Uncertainty Estimation with Probabilistic Neural Networks" by
        Seitzer et al. 2022
        https://openreview.net/forum?id=aPOpXlnV1T
    """

    beta: float = 0.0

    def __call__(
        self, input: torch.distributions.Distribution, target: torch.Tensor
    ) -> torch.Tensor:
        nll = -input.log_prob(target)
        if self.beta > 0.0:
            variance = input.variance
            nll = nll * (variance.detach() ** self.beta)
        return nll


class CRPS(DistributionLoss):
    def __call__(
        self, input: torch.distributions.Distribution, target: torch.Tensor
    ) -> torch.Tensor:
        return input.crps(target)


class EnergyScore(DistributionLoss):
    def __call__(self, input, target: torch.Tensor) -> torch.Tensor:
        return input.energy_score(target)
