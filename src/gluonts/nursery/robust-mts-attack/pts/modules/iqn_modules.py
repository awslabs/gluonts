from math import pi

import torch
from torch import nn as nn


class ImplicitQuantileModule(nn.Module):
    """See arXiv: 1806.06923
    This module, in combination with quantile loss,
    learns how to generate the quantile of the distribution of the target.
    A quantile value, tau, is randomly generated with a Uniform([0, 1])).
    This quantile value is embedded in this module and also passed to the quantile loss:
    this should force the model to learn the appropriate quantile.
    """

    def __init__(self, in_features, output_domain_cls):
        super(ImplicitQuantileModule, self).__init__()
        self.in_features = in_features
        self.quantile_layer = QuantileLayer(in_features)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Softplus(),
            nn.Linear(in_features, 1),
            output_domain_cls(),
        )

    def forward(self, input_data, tau):
        embedded_tau = self.quantile_layer(tau)
        new_input_data = input_data * (torch.ones_like(embedded_tau) + embedded_tau)
        return self.output_layer(new_input_data).squeeze(-1)


class QuantileLayer(nn.Module):
    """Define quantile embedding layer, i.e. phi in the IQN paper (arXiv: 1806.06923)."""

    def __init__(self, num_output):
        super(QuantileLayer, self).__init__()
        self.n_cos_embedding = 64
        self.num_output = num_output
        self.output_layer = nn.Sequential(
            nn.Linear(self.n_cos_embedding, self.n_cos_embedding),
            nn.PReLU(),
            nn.Linear(self.n_cos_embedding, num_output),
        )

    def forward(self, tau):
        cos_embedded_tau = self.cos_embed(tau)
        final_output = self.output_layer(cos_embedded_tau)
        return final_output

    def cos_embed(self, tau):
        integers = torch.repeat_interleave(
            torch.arange(0, self.n_cos_embedding).unsqueeze(dim=0),
            repeats=tau.shape[-1],
            dim=0,
        ).to(tau.device)
        return torch.cos(pi * tau.unsqueeze(dim=-1) * integers)
