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

from typing import List, Optional

import torch

from gluonts.core.component import validated
from gluonts.torch.model.deepvar.module import DeepVARModel
from gluonts.torch.modules.distribution_output import DistributionOutput

from cpflows.flows import ActNorm
from cpflows.icnn import PICNN
from .icnn_utils import DeepConvexNet, SequentialNet


class MQF2MultiItemModel(DeepVARModel):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        target_dim: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        distr_output: DistributionOutput,
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
        icnn_hidden_size: int = 20,
        icnn_num_layers: int = 2,
        is_energy_score: bool = True,
        threshold_input: float = 100,
        es_num_samples: int = 50,
        estimate_logdet: bool = False,
    ) -> None:
        super().__init__(
            freq=freq,
            target_dim=target_dim,
            context_length=context_length,
            prediction_length=prediction_length,
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_real=num_feat_static_real,
            num_feat_static_cat=num_feat_static_cat,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            distr_output=distr_output,
            lags_seq=lags_seq,
            scaling=scaling,
            num_parallel_samples=num_parallel_samples,
        )

        self.threshold_input = threshold_input
        self.es_num_samples = es_num_samples

        convexnet = PICNN(
            dim=target_dim,
            dimh=icnn_hidden_size,
            dimc=hidden_size,
            num_hidden_layers=icnn_num_layers,
            symm_act_first=True,
        )
        deepconvexnet = DeepConvexNet(
            convexnet,
            target_dim,
            is_energy_score=is_energy_score,
            estimate_logdet=estimate_logdet,
        )

        if is_energy_score:
            networks = [deepconvexnet]
        else:
            networks = [
                ActNorm(target_dim),
                deepconvexnet,
                ActNorm(target_dim),
            ]

        self.picnn = SequentialNet(networks)

    @torch.jit.ignore
    def output_distribution(
        self,
        picnn: SequentialNet,
        hidden_state: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        trailing_n: Optional[int] = None,
    ) -> torch.distributions.Distribution:
        if trailing_n is not None:
            hidden_state = hidden_state[:, -trailing_n:]
        return self.distr_output.distribution(picnn, hidden_state, scale=scale)

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        r"""
        For inference, shape:
        feat_static_cat = (batch_size, 1)
        feat_static_real = (batch_size, 1)
        past_time_feat = (batch_size, past_len, num_time_feat)
        past_target = (batch_size, past_len, target_dim)
        past_observed_values = (batch_size, past_len, target_dim)
        future_time_feat = (batch_size, prediction_length, num_time_feat)
        num_parallel_samples: integer
        Return:
        (batch_size, num_parallel_samples, prediction_length, target_dim)
        """

        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        picnn = self.picnn

        _, scale, hidden_state, state = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(
                repeats=self.num_parallel_samples, dim=0
            )
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_state = [
            s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
            for s in state
        ]

        last_hidden_state = hidden_state[:, -1]
        distr = self.output_distribution(picnn, last_hidden_state)

        next_sample = distr.sample(sample_shape=(self.num_parallel_samples,))

        # next_sample shape = (batch_size, num_parallel_samples, 1, target_dim)
        next_sample = next_sample.reshape(
            next_sample.shape[0] * next_sample.shape[1], 1, self.target_dim
        )

        future_samples = [next_sample]

        for k in range(1, self.prediction_length):
            next_features = repeated_time_feat[:, k : k + 1]
            hidden_state, repeated_state = self.lagged_rnn(
                repeated_past_target,
                next_sample,
                next_features,
                repeated_state,
            )

            distr = self.output_distribution(
                picnn, hidden_state.squeeze(dim=-2)
            )
            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample), dim=1
            )
            next_sample = distr.sample().unsqueeze(dim=-2)
            future_samples.append(next_sample)

        unscaled_future_samples = (
            torch.cat(future_samples, dim=-2) * repeated_scale
        )
        return unscaled_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length)
            + self.target_shape,
        )
