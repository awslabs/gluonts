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

# If you use this code in your work please cite:
# Multivariate Time Series Forecasting with Latent Graph Inference
# (https://arxiv.org/abs/2203.03423)


import torch
from torch import nn
from gluonts.torch.modules.distribution_output import StudentTOutput
from typing import Callable
from .networks.model import GNNInfNetwork
from typing import Tuple, Any


def mean_abs_scaling(
    context: torch.Tensor, min_scale: float = 1e-3
) -> torch.Tensor:
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class GNNInfModule(nn.Module):
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: int,
        n_nodes: int,
        distr_output=StudentTOutput(),
        scaling: Callable = mean_abs_scaling,
        gnn_name: str = "gnn",
        nf: int = 64,
        gnn_layers: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        assert prediction_length > 0
        assert context_length > 0
        self.freq = freq
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.scaling = scaling
        self.channels = 48
        self.args_proj = self.distr_output.get_args_proj(self.channels)
        self.nn = GNNInfNetwork(
            in_channels=1,
            out_channels=self.channels,
            input_length=context_length,
            pred_length=prediction_length,
            num_nodes=n_nodes,
            agg_name=gnn_name,
            enc_name="cnnres",
            nf=nf,
            nf_enc=nf,
            gnn_layers=gnn_layers,
        )
        self.device = device
        self.to(device)

    def forward(
        self, context: torch.Tensor
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        context = context.to(self.device)
        bs, context_length, n_nodes = context.size()
        scale = self.scaling(context)
        scaled_context = context / scale
        scaled_context = scaled_context.transpose(1, 2)
        scaled_context = scaled_context.unsqueeze(
            3
        )  # (bs, n_nodes, context_length, in_channels)

        # scaled_context.size -> (bs, n_nodes, context_length, in_channels)
        nn_out = self.nn(scaled_context)
        # nn_out.size -> (bs, n_nodes, pred_length, out_channels)

        nn_out = nn_out.view(
            bs * n_nodes, self.prediction_length, self.channels
        )
        distr_args = self.args_proj(nn_out)
        distr_args = [
            e.reshape(bs, n_nodes, -1).transpose(1, 2) for e in distr_args
        ]
        return distr_args, torch.zeros_like(scale), scale
