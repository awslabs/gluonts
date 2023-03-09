from typing import List

import torch
import torch.nn as nn
from torch.distributions import Distribution

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import DistributionOutput
from gluonts.torch.modules.lambda_layer import LambdaLayer
from pts.modules import MeanScaler, NOPScaler


class SimpleFeedForwardNetworkBase(nn.Module):
    """
    Abstract base class to implement feed-forward networks for probabilistic
    time series prediction.

    This class does not implement hybrid_forward: this is delegated
    to the two subclasses SimpleFeedForwardTrainingNetwork and
    SimpleFeedForwardPredictionNetwork, that define respectively how to
    compute the loss and how to generate predictions.

    Parameters
    ----------
    num_hidden_dimensions
        Number of hidden nodes in each layer.
    prediction_length
        Number of time units to predict.
    context_length
        Number of time units that condition the predictions.
    batch_normalization
        Whether to use batch normalization.
    mean_scaling
        Scale the network input by the data mean and the network output by
        its inverse.
    distr_output
        Distribution to fit.
    kwargs
    """

    @validated()
    def __init__(
        self,
        num_hidden_dimensions: List[int],
        prediction_length: int,
        context_length: int,
        batch_normalization: bool,
        mean_scaling: bool,
        distr_output: DistributionOutput,
    ) -> None:
        super().__init__()

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        self.distr_output = distr_output

        modules = []
        dims = self.num_hidden_dimensions
        for i, units in enumerate(dims[:-1]):
            if i == 0:
                input_size = context_length
            else:
                input_size = dims[i - 1]
            modules += [nn.Linear(input_size, units), nn.ReLU()]
            if self.batch_normalization:
                modules.append(nn.BatchNorm1d(units))
        if len(dims) == 1:
            modules.append(nn.Linear(context_length, dims[-1] * prediction_length))
        else:
            modules.append(nn.Linear(dims[-2], dims[-1] * prediction_length))
        modules.append(
            LambdaLayer(lambda o: torch.reshape(o, (-1, prediction_length, dims[-1])))
        )
        self.mlp = nn.Sequential(*modules)

        self.distr_args_proj = self.distr_output.get_args_proj(dims[-1])

        self.scaler = MeanScaler() if mean_scaling else NOPScaler()

    def get_distr(self, past_target: torch.Tensor) -> Distribution:
        # (batch_size, seq_len, target_dim) and (batch_size, seq_len, target_dim)
        scaled_target, target_scale = self.scaler(
            past_target,
            torch.ones_like(past_target),  # TODO: pass the actual observed here
        )

        mlp_outputs = self.mlp(scaled_target)
        distr_args = self.distr_args_proj(mlp_outputs)
        return self.distr_output.distribution(
            distr_args, scale=target_scale.unsqueeze(1)
        )


class SimpleFeedForwardTrainingNetwork(SimpleFeedForwardNetworkBase):
    def forward(
        self, past_target: torch.Tensor, future_target: torch.Tensor
    ) -> torch.Tensor:
        distr = self.get_distr(past_target)

        # (batch_size, prediction_length, target_dim)
        loss = -distr.log_prob(future_target)

        return loss.mean()


class SimpleFeedForwardPredictionNetwork(SimpleFeedForwardNetworkBase):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        distr = self.get_distr(past_target)

        # (num_samples, batch_size, prediction_length)
        samples = distr.sample((self.num_parallel_samples,))

        return samples.permute(1, 0, 2)
