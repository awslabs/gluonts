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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.util import lagged_sequence_values


class DeepARModel(nn.Module):
    """
    Module implementing the DeepAR model, see [SFG17]_.

    *Note:* the code of this model is unrelated to the implementation behind
    `SageMaker's DeepAR Forecasting Algorithm
    <https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html>`_.

    Parameters
    ----------
    freq
        String indicating the sampling frequency of the data to be processed.
    context_length
        Length of the RNN unrolling prior to the forecast date.
    prediction_length
        Number of time points to predict.
    num_feat_dynamic_real
        Number of dynamic real features that will be provided to ``forward``.
    num_feat_static_real
        Number of static real features that will be provided to ``forward``.
    num_feat_static_cat
        Number of static categorical features that will be provided to
        ``forward``.
    cardinality
        List of cardinalities, one for each static categorical feature.
    embedding_dimension
        Dimension of the embedding space, one for each static categorical
        feature.
    num_layers
        Number of layers in the RNN.
    hidden_size
        Size of the hidden layers in the RNN.
    dropout_rate
        Dropout rate to be applied at training time.
    distr_output
        Type of distribution to be output by the model at each time step
    lags_seq
        Indices of the lagged observations that the RNN takes as input. For
        example, ``[1]`` indicates that the RNN only takes the observation at
        time ``t-1`` to produce the output for time ``t``; instead,
        ``[1, 25]`` indicates that the RNN takes observations at times ``t-1``
        and ``t-25`` as input.
    scaling
        Whether to apply mean scaling to the observations (target).
    num_parallel_samples
        Number of samples to produce when unrolling the RNN in the prediction
        time range.
    """

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
        self.past_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
        self.rnn_input_size = len(self.lags_seq) + self._number_of_features
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
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

    def input_shapes(self, batch_size=1) -> Dict[str, Tuple[int, ...]]:
        return {
            "feat_static_cat": (batch_size, self.num_feat_static_cat),
            "feat_static_real": (batch_size, self.num_feat_static_real),
            "past_time_feat": (
                batch_size,
                self._past_length,
                self.num_feat_dynamic_real,
            ),
            "past_target": (batch_size, self._past_length) + self.target_shape,
            "past_observed_values": (batch_size, self._past_length),
            "future_time_feat": (
                batch_size,
                self.prediction_length,
                self.num_feat_dynamic_real,
            ),
        }

    def input_types(self) -> Dict[str, torch.dtype]:
        return {
            "feat_static_cat": torch.long,
            "feat_static_real": torch.float,
            "past_time_feat": torch.float,
            "past_target": torch.float,
            "past_observed_values": torch.float,
            "future_time_feat": torch.float,
        }

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
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Applies the underlying RNN to the provided target data and covariates.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length, *target_shape)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        future_target
            (Optional) tensor of future target values,
            shape: ``(batch_size, prediction_length, *target_shape)``.

        Returns
        -------
        Tuple
            A tuple containing, in this order:
            - Parameters of the output distribution
            - Scaling factor applied to the target
            - Raw output of the RNN
            - Static input to the RNN
            - Output state from the RNN
        """
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        prior_input = past_target[:, : -self.context_length] / scale
        input = (
            torch.cat((context, future_target[:, :-1]), dim=1) / scale
            if future_target is not None
            else context / scale
        )

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, input.shape[1], -1
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
        lags = lagged_sequence_values(self.lags_seq, prior_input, input)
        rnn_input = torch.cat((lags, features), dim=-1)

        output, new_state = self.rnn(rnn_input)

        params = self.param_proj(output)
        return params, scale, output, static_feat, new_state

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        """
        Instantiate the output distribution

        Parameters
        ----------
        params
            Tuple of distribution parameters.
        scale
            (Optional) scale tensor.
        trailing_n
            If set, the output distribution is created only for the last
            ``trailing_n`` time points.

        Returns
        -------
        torch.distributions.Distribution
            Output distribution from the model.
        """
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
        """
        Invokes the model on input data, and produce outputs future samples.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length, *target_shape)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        num_parallel_samples
            How many future sampels to produce.
            By default, self.num_parallel_samples is used.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        params, scale, _, static_feat, state = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_scale = scale.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_state = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=1)
            for s in state
        ]

        repeated_params = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=0)
            for s in params
        ]
        distr = self.output_distribution(
            repeated_params, trailing_n=1, scale=repeated_scale
        )
        next_sample = distr.sample()
        future_samples = [next_sample]

        for k in range(1, self.prediction_length):
            scaled_next_sample = next_sample / repeated_scale
            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )
            next_lags = lagged_sequence_values(
                self.lags_seq,
                repeated_past_target,
                scaled_next_sample,
            )
            rnn_input = torch.cat((next_lags, next_features), dim=-1)

            output, repeated_state = self.rnn(rnn_input, repeated_state)

            repeated_past_target = torch.cat(
                (repeated_past_target, scaled_next_sample), dim=1
            )

            params = self.param_proj(output)
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            future_samples.append(next_sample)

        future_samples_concat = torch.cat(future_samples, dim=1)

        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length)
            + self.target_shape,
        )
