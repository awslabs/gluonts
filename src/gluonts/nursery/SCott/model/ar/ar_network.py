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

from typing import List

import torch
import torch.nn as nn


class ARNetworkBase(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
    ) -> None:
        super().__init__()

        self.prediction_length = prediction_length
        self.context_length = context_length
        # self.criterion = nn.MSELoss(reduction='none')
        self.criterion = nn.SmoothL1Loss(reduction="none")

        modules = []
        modules.append(nn.Linear(context_length, prediction_length))
        self.linear = nn.Sequential(*modules)


class ARTrainingNetwork(ARNetworkBase):
    def forward(
        self, past_target: torch.Tensor, future_target: torch.Tensor
    ) -> torch.Tensor:
        # m = max(torch.mean(past_target), torch.mean(future_target))
        # past_target /= m
        # future_target /= m
        nu = min(
            torch.mean(past_target).item(), torch.mean(future_target).item()
        )
        past_target /= 1 + nu
        future_target /= 1 + nu
        prediction = self.linear(past_target)
        loss = self.criterion(prediction, future_target)
        return loss


class ARPredictionNetwork(ARNetworkBase):
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        pass
