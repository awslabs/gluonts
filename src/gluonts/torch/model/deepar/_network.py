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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.util import weighted_average


class DeepARNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "LSTM",
        context_length: Optional[int] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        dropout_rate: float = 0.1,
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.context_length = context_length or prediction_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = cardinality
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or self.cardinality is None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )
        self.scaling = scaling
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.history_length = self.context_length + max(self.lags_seq)
        # for decoding the lags are shifted by one, at the first time-step
        # of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]
        self.distr_output = distr_output
        if self.cell_type == "LSTM":
            rnn_type = nn.LSTM
        elif self.cell_type == "GRU":
            rnn_type = nn.GRU
        self.rnn = rnn_type(
            input_size=self._rnn_input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.target_shape = distr_output.event_shape
        self.proj_distr_args = distr_output.get_args_proj(num_cells)
        self.embedder = FeatureEmbedder(
            cardinalities=self.cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

    @property
    def _rnn_input_size(self) -> int:
        return (
            1  # TODO adjust this for multivariate target?
            + len(self.lags_seq)
            + sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.

        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        indices : List[int]
            list of lag indices to be used.
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

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def unroll_encoder(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, Union[torch.Tensor, List], torch.Tensor, torch.Tensor
    ]:
        if future_target is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:, -self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            sequence = torch.cat((past_target, future_target), dim=1)
            subsequences_length = self.context_length + self.prediction_length

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )
        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                scale.log(),
            ),
            dim=1,
        )
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )
        lags_scaled = lags / scale.unsqueeze(-1)

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = lags_scaled.reshape(
            (
                -1,
                subsequences_length,
                len(self.lags_seq) * prod(self.target_shape),
            )
        )

        inputs = torch.cat(
            (input_lags, time_feat, repeated_static_feat), dim=-1
        )
        outputs, state = self.rnn(inputs)

        # outputs: (batch_size, seq_len, num_cells)
        # state: list of (num_layers, batch_size, num_cells) tensors
        # scale: (batch_size, 1, *target_shape)
        # static_feat: (batch_size, num_features + prod(target_shape))
        return outputs, state, scale, static_feat

    def sampling_decoder(
        self,
        static_feat: torch.Tensor,
        past_target: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states,
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch. Shape: (batch_size, 1, 1).
        begin_states : List or Tensor
            list of initial states for the LSTM layers or tensor for GRU.
            the shape of each tensor of the list should be (num_layers, batch_size, num_cells)
        Returns
        --------
        Tensor
            A tensor containing sampled paths.
            Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size * self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_time_feat = time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        ).unsqueeze(1)
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_states = (
            [
                s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
                for s in begin_states
            ]
            if self.cell_type == "LSTM"
            else begin_states.repeat_interleave(
                repeats=self.num_parallel_samples, dim=1
            )
        )

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and update the state
        for k in range(self.prediction_length):
            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = lags / repeated_scale.unsqueeze(-1)

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = lags_scaled.reshape(
                (-1, 1, prod(self.target_shape) * len(self.lags_seq))
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            decoder_input = torch.cat(
                (
                    input_lags,
                    repeated_time_feat[:, k : k + 1, :],
                    repeated_static_feat,
                ),
                dim=-1,
            )

            # output shape: (batch_size * num_samples, 1, num_cells)
            # state shape: (batch_size * num_samples, num_cells)
            rnn_outputs, repeated_states = self.rnn(
                decoder_input, repeated_states
            )

            distr_args = self.proj_distr_args(rnn_outputs)

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(
                distr_args, scale=repeated_scale
            )

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample()

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = torch.cat(
                (repeated_past_target, new_samples), dim=1
            )
            future_samples.append(new_samples)

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, *target_shape)
        return samples.reshape(
            (
                (-1, self.num_parallel_samples)
                + (self.prediction_length,)
                + self.target_shape
            )
        )

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
        sampling: Optional[bool] = None,
    ) -> Union[torch.Tensor, torch.distributions.Distribution]:
        rnn_outputs, state, scale, static_feat = self.unroll_encoder(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        if sampling is False or (sampling is None and self.training):
            distr_args = self.proj_distr_args(rnn_outputs)
            return self.distr_output.distribution(distr_args, scale=scale)

        return self.sampling_decoder(
            past_target=past_target,
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            begin_states=state,
        )


class DeepARLightningNetwork(DeepARNetwork, pl.LightningModule):
    def __init__(
        self,
        *args,
        loss: DistributionLoss = NegativeLogLikelihood(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.optimizer = optimizer or torch.optim.Adam(
            self.parameters(), lr=1e-3
        )

    def _compute_loss(
        self,
        distr,
        past_target,
        future_target,
        past_observed_values,
        future_observed_values,
    ):
        context_target = past_target[:, -self.context_length :, ...]
        target = torch.cat((context_target, future_target), dim=1)
        loss = self.loss(distr, target)

        context_observed = past_observed_values[:, -self.context_length :, ...]
        observed_values = torch.cat(
            (context_observed, future_observed_values), dim=1
        )

        if len(self.target_shape) == 0:
            loss_weights = observed_values
        else:
            loss_weights = observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss, weights=loss_weights)

    def _loss_step(self, batch):
        distr = self(
            batch["feat_static_cat"],
            batch["feat_static_real"],
            batch["past_time_feat"],
            batch["past_target"],
            batch["past_observed_values"],
            batch["future_time_feat"],
            batch["future_target"],
            sampling=False,
        )
        return self._compute_loss(
            distr,
            batch["past_target"],
            batch["future_target"],
            batch["past_observed_values"],
            batch["future_observed_values"],
        )

    def training_step(self, batch, *args, **kwargs):
        """Execute training step"""
        return self._loss_step(batch)

    def validation_step(self, batch, *args, **kwargs):
        """Execute validation step"""
        return self._loss_step(batch)

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        return self.optimizer
