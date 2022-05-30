from typing import List

import torch
import torch.nn as nn


class LSTMNetworkBase(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        input_size: int = 1,
        hidden_layer_size: int = 100,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.prediction_length = prediction_length
        self.context_length = context_length
        # self.criterion = nn.MSELoss(reduction='none')
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)
        self.linear = nn.Linear(hidden_layer_size, prediction_length)

        # self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
        #                    torch.zeros(1,1,self.hidden_layer_size))


class LSTMTrainingNetwork(LSTMNetworkBase):
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
        # import pdb;pdb.set_trace()

        inputs = past_target.view(
            past_target.shape[1], past_target.shape[0], 1
        )
        lstm_out, _ = self.lstm(inputs)
        prediction = self.linear(lstm_out)

        loss = self.criterion(prediction[-1], future_target)
        return loss


class LSTMPredictionNetwork(LSTMNetworkBase):
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        pass
