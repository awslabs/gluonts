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

from turtle import distance
from typing import List, Optional, Tuple, Union
from matplotlib.pyplot import sca

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import DistributionOutput
from pts.model import weighted_average
from pts.modules import MeanScaler, NOPScaler, FeatureEmbedder
from torch.distributions import (
    AffineTransform,
    Beta,
    Distribution,
    Gamma,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    TransformedDistribution,
)


class DeepVARTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        cardinality: List[int],
        embedding_dimension: List[int],
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.target_dim = target_dim
        self.scaling = scaling
        self.target_dim_sample = target_dim

        assert len(set(lags_seq)) == len(
            lags_seq
        ), "no duplicated lags allowed!"
        lags_seq.sort()

        self.lags_seq = lags_seq

        self.distr_output = distr_output

        self.target_dim = target_dim

        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell_type]
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.target_shape = distr_output.event_shape
        self.proj_dist_args = distr_output.get_args_proj(num_cells)

        self.embed = FeatureEmbedder(
            cardinalities=cardinality, embedding_dims=embedding_dimension
        )

        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(
                sequence[:, begin_index:end_index, ...].unsqueeze(1)
            )
        if torch.cat(lagged_values, dim=1).ndim > 4:
            return (
                torch.cat(lagged_values, dim=1)
                .view(torch.cat(lagged_values, dim=1).shape[:4])
                .permute(0, 2, 3, 1)
            )
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        unroll_length: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        # assert_shape(
        #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        # )
        input_lags = lags_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        )

        # (batch_size, target_dim, embed_dim)
        embedded_cat = self.embed(feat_static_cat)
        # assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                scale.log()
                if len(self.target_shape) == 0
                else scale.squeeze(1).log(),
            ),
            dim=1,
        )

        # (batch_size, seq_len, embed_dim)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, unroll_length, -1
        )

        # (batch_size, sub_seq_len, input_dim)

        inputs = torch.cat(
            (input_lags, repeated_static_feat, time_feat), dim=-1
        )

        # unroll encoder

        outputs, state = self.rnn(inputs, begin_state)

        # assert_shape(outputs, (-1, unroll_length, self.num_cells))
        # for s in state:
        #     assert_shape(s, (-1, self.num_cells))

        # assert_shape(
        #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        # )

        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        feat_static_cat: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unrolls the RNN encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of
        past_target_cdf and a vector of static features that was constructed
        and fed as input to the encoder. All tensor arguments should have NTC
        layout.

        Parameters
        ----------
        feat_static_real

        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        feat_static_cat


        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        scale
            Mean scales for the time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN

        """
        # print(past_observed_values.shape)
        # print(past_is_pad.unsqueeze(-1).shape)
        if past_observed_values.ndim == 4:
            past_observed_values = past_observed_values.view(
                past_observed_values.shape[:3]
            )
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:, -self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        outputs, states, lags_scaled, inputs = self.unroll(
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            unroll_length=subsequences_length,
            begin_state=None,
        )

        return outputs, states, scale, lags_scaled, inputs

    def distr(
        self,
        rnn_outputs: torch.Tensor,
        scale: torch.Tensor,
    ):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        distr_args = self.proj_dist_args(rnn_outputs)

        # compute likelihood of target given the predicted parameters
        distr = self.distr_output.distribution(distr_args, scale=scale)
        return distr, distr_args

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        feat_static_cat

        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """

        seq_len = self.context_length + self.prediction_length

        # unroll the decoder in "training mode", i.e. by providing future data
        # as well
        rnn_outputs, _, scale, _, inputs = self.unroll_encoder(
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            feat_static_cat=feat_static_cat,
        )

        # put together target sequence
        # (batch_size, seq_len, target_dim)
        target = torch.cat(
            (
                past_target_cdf[:, -self.context_length :, ...],
                future_target_cdf,
            ),
            dim=1,
        )

        # assert_shape(target, (-1, seq_len, self.target_dim))

        distr, distr_args = self.distr(rnn_outputs=rnn_outputs, scale=scale)
        # self.distr._validate_args = False

        # we sum the last axis to have the same shape for all likelihoods
        # (batch_size, subseq_length, 1)
        likelihoods = -distr.log_prob(target).unsqueeze(-1)

        # assert_shape(likelihoods, (-1, seq_len, 1))

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # (batch_size, subseq_length, target_dim)
        observed_values = torch.cat(
            (
                past_observed_values[:, -self.context_length :, ...],
                future_observed_values,
            ),
            dim=1,
        )

        # mask the loss at one time step if one or more observations is missing
        # in the target dimensions (batch_size, subseq_length, 1)
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)

        # assert_shape(loss_weights, (-1, seq_len, 1))

        loss = weighted_average(likelihoods, weights=loss_weights, dim=1)

        # assert_shape(loss, (-1, -1, 1))
        return (loss.mean(), likelihoods) + distr_args + (target,)


class DeepVARPredictionNetwork(DeepVARTrainingNetwork):
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        feat_static_cat

        feat_static_real

        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
        """

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(
                repeats=self.num_parallel_samples, dim=dim
            )

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        repeated_feat_static_cat = repeat(feat_static_cat)
        repeated_feat_static_real = repeat(feat_static_real)

        # slight difference for GPVAR and DeepVAR, in GPVAR, its a list
        if self.cell_type == "LSTM":
            repeated_states = [repeat(s, dim=1) for s in begin_states]
        else:
            repeated_states = repeat(begin_states, dim=1)

        future_samples = []

        # for each future time-units we draw new samples for this time-unit
        # and update the state
        loc = []
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, _, _ = self.unroll(
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat[:, k : k + 1, ...],
                feat_static_cat=repeated_feat_static_cat,
                feat_static_real=repeated_feat_static_real,
                unroll_length=1,
            )

            distr, distr_args = self.distr(
                rnn_outputs=rnn_outputs,
                scale=repeated_scale,
            )
            # print(distr_args[0].shape) 700 x 1 x 10
            # print(distr_args[1].shape) 700 x 1 x 10 x 10
            # (batch_size, 1, target_dim)
            new_samples = distr.rsample()
            loc.append(distr.base_dist.loc)
            # transforms = [
            #         AffineTransform(
            #             loc=0.0 if loc is None else loc,
            #             scale=repeated_scale,
            #         )
            #     ]
            # for transform in transforms:
            #     loc = transform(loc)

            # (batch_size, seq_len, target_dim)
            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, new_samples), dim=1
            )

        # (batch_size * num_samples, prediction_length, target_dim)
        samples = torch.cat(future_samples, dim=1)
        locs = torch.cat(loc, dim=1)
        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )  # , locs.reshape(
        # -1,
        #     self.num_parallel_samples,
        #     self.prediction_length,
        #     self.target_dim,
        # )

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------
        feat_static_cat

        feat_static_real

        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)

        Returns
        -------
        sample_paths : Tensor
            A tensor containing sampled paths (1, num_sample_paths,
            prediction_length, target_dim).
        mean: mean of the produced distribution
        scale: scale factor

        """

        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        # print((past_is_pad.unsqueeze(-1)).shape)
        # if past_observed_values.ndim == 4:
        #     past_observed_values = past_observed_values.view(past_observed_values.shape[:3])
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, state, scale, _, _ = self.unroll_encoder(
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            feat_static_cat=feat_static_cat,
        )
        samples = self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            time_feat=future_time_feat,
            scale=scale,
            begin_states=state,
        )
        return samples  # , scale
