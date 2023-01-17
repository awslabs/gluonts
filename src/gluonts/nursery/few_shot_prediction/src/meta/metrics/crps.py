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


class CRPS(Metric):
    """
    Same as mean_weighted_quantile_loss in meta.evaluation.metrics just for pytorch

    Parameters
    ----------
    quantiles: The quantiles.
    """

    def __init__(
        self,
        quantiles: List[str],
        rescale: bool = False,
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
        self.rescale = rescale
        self.add_state(
            "quantile_losses",
            default=torch.as_tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "denom", default=torch.as_tensor(0.0), dist_reduce_fx="sum"
        )

    def update(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
    ) -> None:
        if self.rescale:
            m = scales[:, 0].unsqueeze(1)
            std = scales[:, 1].unsqueeze(1)
            y_true = y_true * std + m
            y_pred = y_pred * std.unsqueeze(2) + m.unsqueeze(2)

        # crop y_pred to shape of y_true since the model always predicts the max prediction length
        # max prediction length might be longer than max length in the particular batch
        y_pred = y_pred[:, : y_true.size()[1], ...]
        y_true = y_true.unsqueeze(-1)
        quantile_losses = 2 * torch.sum(
            torch.abs(
                (y_pred - y_true)
                * ((y_true <= y_pred).type(torch.uint8) - self.quantiles)
            ),
            axis=-1,
        )  # shape [num_time_series, max_ts_length]
        self.denom += torch.sum(torch.abs(y_true))
        # mask out all elements that correspond to padding in y_true
        quantile_losses = torch.mul(quantile_losses, mask)
        self.quantile_losses += quantile_losses.sum() / len(self.quantiles)

    def compute(self) -> torch.Tensor:
        return self.quantile_losses / self.denom
