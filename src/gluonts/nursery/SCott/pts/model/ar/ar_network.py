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
