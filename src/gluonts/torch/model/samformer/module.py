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

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.util import weighted_average


class SamFormerModel(nn.Module):
    """
    Module implementing the SamFormer model for multivariate forecasting as
    described in TODO extended to be probabilistic.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    hidden_dim
        Dim of query and key projection.
    scaling
        Whether to scale the input using mean or std or None.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    nonnegative_pred_samples
        Should final prediction samples be non-negative? If yes, an activation
        function is applied to ensure non-negative. Observe that this is applied
        only to the final samples and this is not applied during training.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dim: int,
        scaling: Optional[str],
        distr_output=StudentTOutput(),
        nonnegative_pred_samples: bool = False,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dim = hidden_dim

        self.distr_output = distr_output

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)
        self.nonnegative_pred_samples = nonnegative_pred_samples

        self.compute_keys = nn.Linear(context_length + 2, hidden_dim)
        self.compute_queries = nn.Linear(context_length + 2, hidden_dim)
        self.compute_values = nn.Linear(context_length + 2, context_length)

        # project each variate to prediction length number of latent variables
        self.projection = nn.Linear(
            context_length, prediction_length * hidden_dim
        )

        # project each prediction length latent to distribution parameters
        self.args_proj = self.distr_output.get_args_proj(hidden_dim)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length, -1),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length, -1),
                    dtype=torch.float,
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )
        log_abs_loc = loc.sign() * loc.abs().log1p()
        log_scale = scale.log()

        # Transpose to time last
        past_target_scaled = past_target_scaled.transpose(1, 2)
        log_abs_loc = log_abs_loc.transpose(1, 2)
        log_scale = log_scale.transpose(1, 2)

        # concatenate past target with log_abs_loc and log_scale
        expanded_target_scaled = torch.cat(
            [past_target_scaled, log_abs_loc, log_scale], dim=-1
        )

        queries = self.compute_queries(expanded_target_scaled)
        keys = self.compute_keys(expanded_target_scaled)
        values = self.compute_values(expanded_target_scaled)
        att_score = F.scaled_dot_product_attention(queries, keys, values)
        out = past_target_scaled + att_score

        # project to prediction length * hidden_dim and reshape
        projection_out = self.projection(out).reshape(
            -1,
            past_target.shape[2],
            self.prediction_length,
            self.hidden_dim,
        )

        # transpose to prediction length first
        projection_out_transpose = projection_out.transpose(1, 2)

        # project to distribution arguments
        distr_args = self.args_proj(projection_out_transpose)
        return distr_args, loc, scale

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        distr_args, loc, scale = self(
            past_target=past_target, past_observed_values=past_observed_values
        )
        loss = self.distr_output.loss(
            target=future_target, distr_args=distr_args, loc=loc, scale=scale
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)
