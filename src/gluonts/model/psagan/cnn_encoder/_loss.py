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
import torch.nn as nn
import torch.nn.functional as F

from gluonts.core.component import validated


class SimilarityScore(nn.Module):
    @validated()
    def __init__(self):
        super(SimilarityScore, self).__init__()

    def forward(self, x1, x2):
        """
        - Input:
            * x1: float tensor of shape (batch_size, nb_sample, length)
            * x2: float tensor of shape (batch_size, nb_sample, length)
        - Output:
            * float tensor of shape (batch_size, 1)
        """
        assert (
            x1.shape == x2.shape
        ), "x1 and x2 must be of same shape: (batch_size, nb_sample, length)"
        batch_size = x1.size(0)
        nb_sample = x1.size(1)
        score = (
            F.logsigmoid(
                torch.matmul(x1.unsqueeze(-2), x2.unsqueeze(-1)).view(
                    batch_size, nb_sample
                )
            )
            .mean(axis=1)
            .view(-1, 1)
        )

        return score


class TripletLoss(nn.Module):
    @validated()
    def __init__(self, score: SimilarityScore = SimilarityScore()):
        super(TripletLoss, self).__init__()
        self.score = score

    def forward(self, x_ref, x_pos, x_neg):
        """
        - Input:
            *x_ref: float tensor (batch_size, 1, embedding_dimension)
            *x_pos: float tensor (batch_size, 1, embedding_dimension)
            *x_neg: float tensor (batch_size, nb_negative_samples, embedding_dimension)
        """

        loss = -self.score(x_pos, x_ref) - self.score(
            -x_ref.expand_as(x_neg), x_neg
        )

        return loss
