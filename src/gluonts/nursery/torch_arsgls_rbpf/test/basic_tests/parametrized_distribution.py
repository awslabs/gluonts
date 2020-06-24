import torch
from torch.distributions import MultivariateNormal, Distribution
from torch import nn
from torch_extensions.distributions.conditional_parametrised_distribution import (
    ParametrisedConditionalDistribution,
)
from utils.utils import Bias, BatchDiagMatrix


def test_parametrized_distribution():
    class Dims:
        timesteps = 10
        particle = 20
        batch = 30
        outputs = 40
        inputs = 50
        hidden_stem = 60

    dims = Dims

    inpt = torch.randn(dims.timesteps, dims.batch, dims.inputs)
    model = ParametrisedConditionalDistribution(
        stem=nn.Sequential(
            nn.Linear(in_features=dims.inputs, out_features=dims.hidden_stem),
            nn.ReLU(),
            nn.Linear(
                in_features=dims.hidden_stem, out_features=dims.hidden_stem
            ),
            nn.ReLU(),
        ),
        dist_params=nn.ModuleDict(
            {
                "loc": nn.Linear(
                    in_features=dims.hidden_stem, out_features=dims.outputs
                ),
                "scale_tril": nn.Sequential(
                    Bias(dim_out=dims.outputs, init_val=0.1),
                    nn.Softplus(),
                    BatchDiagMatrix(),
                ),
            }
        ),
        dist_cls=MultivariateNormal,
    )
    dist = model(inpt)
    assert isinstance(dist, Distribution)
    assert dist.sample().shape == (dims.timesteps, dims.batch, dims.outputs)
