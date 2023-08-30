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

from typing import Any, Callable, Optional
import torch
from torchmetrics import Metric


class NormalizedDeviation(Metric):
    # pylint: disable=arguments-differ

    def __init__(
        self,
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
        self.rescale = rescale
        self.add_state(
            "numerator", default=torch.as_tensor(0.0), dist_reduce_fx="sum"
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

        idx_median = y_pred.size()[-1] // 2
        y_pred = y_pred[:, : y_true.size()[1], idx_median]
        losses = (y_pred - y_true).abs()
        losses = torch.mul(losses, mask)

        self.numerator += losses.sum()
        self.denom += y_true.abs().sum()

    def compute(self) -> torch.Tensor:
        return self.numerator / self.denom
