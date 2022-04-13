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

from typing import Literal, Optional
import torch
from torch import nn


class ListMLELoss(nn.Module):
    """
    Loss that is used for ListMLE.

    For each feature, ranking is performed independently. A lower score
    indicates a lower rank (i.e. "better" value).
    """

    def __init__(
        self,
        discount: Optional[
            Literal["logarithmic", "linear", "quadratic"]
        ] = None,
    ):
        """
        Args:
            discount: The type of discount to apply. If discounting, higher-ranked values are more
                important.
        """
        super().__init__()
        self.discount = discount

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the listwise loss of the predictions.

        Args:
            y_pred: Tensor of shape [N, D] with the predicted values (N: number of items, D:
                number of metrics).
            y_true: Tensor of shape [N, D] with the actual values.
            group_ids: Tensor of shape [N] with the group IDs for each item. All items within a
                group are ranked.

        Returns:
            Tensor of shape [1] containing the loss.
        """
        # We compute the loss for each group and subsequently average
        log_likelihoods = []
        for group_id in group_ids.unique():
            # First, we extract the values belonging to the group
            mask = group_ids == group_id
            group_pred = -y_pred[
                mask
            ]  # for ListMLE, a higher score is "better, so we invert here
            group_true = y_true[mask]

            # Then, compute numerator and denominator
            order = group_true.argsort(0)
            num = group_pred.gather(0, order)
            denom = num.flip(0).logcumsumexp(0).flip(0)

            # And compute the log likelihood
            log_likeli = num - denom

            # Optionally, we also apply the discount factor
            n = log_likeli.size(0)
            if self.discount is not None:
                if self.discount == "logarithmic":
                    denom = (torch.arange(n) + 2)[:, None].log()
                elif self.discount == "linear":
                    denom = (torch.arange(n) + 1)[:, None]
                else:  # self.discount == "quadratic"
                    denom = (torch.arange(n) + 1)[:, None] ** 2
                # We additionally scale by n / denom.sum() to scale the loss
                log_likeli = (log_likeli / denom) * (
                    n / denom.reciprocal().sum()
                )

            log_likelihoods.append(log_likeli.mean())

        # Eventually compute the loss as the NLL
        loss = -torch.stack(log_likelihoods).mean()
        return loss
