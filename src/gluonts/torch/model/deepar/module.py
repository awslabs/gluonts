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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder


class DeepARModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(hidden_size)
        self.target_shape = distr_output.event_shape
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.history_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
        self.lagged_rnn = LaggedLSTM(
            input_size=1,  # TODO fix
            features_size=self._number_of_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            lags_seq=[lag - 1 for lag in self.lags_seq],
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + 1  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def unroll_lagged_rnn(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        prior_input = past_target[:, : -self.context_length] / scale
        input = (
            torch.cat((context, future_target[:, :-1]), dim=1) / scale
            if future_target is not None
            else context / scale
        )

        unroll_length = (
            self.context_length
            if future_target is None
            else self.context_length + future_target.shape[1] - 1
        )
        assert input.shape[1] == unroll_length

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, unroll_length, -1
        )

        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, -self.context_length + 1 :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_time_feat is not None
            else past_time_feat[:, -self.context_length + 1 :, ...]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        output, new_state = self.lagged_rnn(prior_input, input, features)

        params = self.param_proj(output)
        return params, scale, static_feat, new_state

    @torch.jit.ignore
    def output_distribution(
        self, params, scale, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)

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
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        params, scale, static_feat, state = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_static_feat = static_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = past_target.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_state = [
            s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
            for s in state
        ]

        distr = self.output_distribution(params, scale, trailing_n=1)

        new_sample = distr.sample(sample_shape=(self.num_parallel_samples,))
        new_sample = new_sample.reshape(
            (new_sample.shape[0] * new_sample.shape[1], -1)
        )
        all_samples = [new_sample]

        for k in range(1, self.prediction_length):
            new_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )
            output, repeated_state = self.lagged_rnn(
                repeated_past_target, new_sample, new_features, repeated_state
            )
            params = self.param_proj(output)
            distr = self.output_distribution(params, repeated_scale)
            repeated_past_target = torch.cat(
                (repeated_past_target, new_sample), dim=1
            )
            new_sample = distr.sample()
            all_samples.append(new_sample)

        repeated_past_target = torch.cat(
            (repeated_past_target, new_sample), dim=1
        )

        return torch.reshape(
            repeated_past_target[:, -self.prediction_length :],
            (-1, self.num_parallel_samples, self.prediction_length),
        )


class LaggedLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        features_size: int,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        lags_seq: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.features_size = features_size
        self.dropout_rate = dropout_rate
        self.lags_seq = lags_seq
        self.rnn_input_size = input_size * len(self.lags_seq) + features_size
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

    def get_lagged_subsequences(
        self,
        sequence: torch.Tensor,
        subsequences_length: int,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.

        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.

        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = self.lags_seq

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def _check_shapes(
        self,
        prior_input: torch.Tensor,
        input: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> None:
        assert len(prior_input.shape) == len(input.shape)
        assert (
            len(prior_input.shape) == 2 and self.input_size == 1
        ) or prior_input.shape[2] == self.input_size
        assert (len(input.shape) == 2 and self.input_size == 1) or input.shape[
            -1
        ] == self.input_size
        assert (
            features is None or features.shape[2] == self.features_size
        ), f"{features.shape[2]}, expected {self.features_size}"

    def forward(
        self,
        prior_input: torch.Tensor,
        input: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self._check_shapes(prior_input, input, features)

        sequence = torch.cat((prior_input, input), dim=1)
        lagged_sequence = self.get_lagged_subsequences(
            sequence=sequence,
            subsequences_length=input.shape[1],
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], -1
        )

        if features is None:
            rnn_input = reshaped_lagged_sequence
        else:
            rnn_input = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        if state is None:
            return self.rnn(rnn_input)
        else:
            return self.rnn(rnn_input, state)
