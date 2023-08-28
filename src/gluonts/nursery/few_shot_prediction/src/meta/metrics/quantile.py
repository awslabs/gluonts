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

from typing import List, Optional, Callable, Any
import torch
from torchmetrics import Metric


class QuantileLoss(Metric):
    """
    Computes the quantile loss.

    Parameters
    ----------
    quantiles: The quantiles.
    """

    def __init__(
        self,
        quantiles: List[str],
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.register_buffer(
            "quantiles", torch.as_tensor([float(q) for q in quantiles])
        )
        self.add_state(
            "quantile_sum", default=torch.as_tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_observations",
            default=torch.as_tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor
    ) -> None:
        # crop y_pred to shape of y_true since the model always predicts the max prediction length
        # max prediction length might be longer than max length in the particular batch
        y_pred = y_pred[:, : y_true.size()[1], ...]
        y_true = y_true.unsqueeze(-1)
        losses = torch.sum(
            torch.abs(
                (y_pred - y_true)
                * ((y_true <= y_pred).type(torch.uint8) - self.quantiles)
            ),
            axis=-1,
        )
        # mask out all elements that correspond to padding in y_true
        losses = torch.mul(losses, mask)
        self.quantile_sum += losses.sum()
        self.num_observations += mask.sum() * len(self.quantiles)

    def compute(self) -> torch.Tensor:
        return self.quantile_sum / self.num_observations
