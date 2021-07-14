# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from torch import nn


class ContrastiveClasifier(nn.Module):
    """Contrastive Classifier.

    Calculates the distance between two random vectors, and returns an exponential transformation of it,
    which can be interpreted as the logits for the two vectors being different.

    p : Probability of x1 and x2 being different

    p = 1 - exp( -dist(x1,x2) )
    """

    def __init__(
        self,
        distance: nn.Module,
    ):
        """
        Args:
            distance : A Pytorch module which takes two (batches of) vectors and returns a (batch of)
                positive number.
        """
        super().__init__()

        self.distance = distance

        self.eps = 1e-10

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:

        # Compute distance
        dists = self.distance(x1, x2)

        # Probability of the two embeddings being equal: exp(-dist)
        log_prob_equal = -dists

        # Computation of log_prob_different
        prob_different = torch.clamp(1 - torch.exp(log_prob_equal), self.eps, 1)
        log_prob_different = torch.log(prob_different)

        logits_different = log_prob_different - log_prob_equal

        return logits_different
