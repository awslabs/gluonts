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

import mxnet as mx

from gluonts.core.component import validated
from gluonts.model.transformer.trans_decoder import TransformerDecoder
from gluonts.model.transformer.trans_encoder import TransformerEncoder
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import DistributionOutput

LARGE_NEGATIVE_VALUE = -99999999


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class TransformerNetwork(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        cardinality: List[int],
        embedding_dimension: int,
        lags_seq: List[int],
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.scaling = scaling
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.distr_output = distr_output

        assert len(set(lags_seq)) == len(
            lags_seq
        ), "no duplicated lags allowed!"
        lags_seq.sort()

        self.lags_seq = lags_seq

        self.target_shape = distr_output.event_shape

        with self.name_scope():
            self.proj_dist_args = distr_output.get_args_proj()
            self.encoder = encoder
            self.decoder = decoder
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=[embedding_dimension for _ in cardinality],
            )

            if scaling:
                self.scaler = MeanScaler(keepdims=True)
            else:
                self.scaler = NOPScaler(keepdims=True)

    @staticmethod
    def get_lagged_subsequences(
        F,
        sequence: Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> Tensor:
        """
        Returns lagged subsequences of a given sequence.

        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted. Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.

        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and I = len(indices), containing lagged
            subsequences. Specifically, lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: sequence_length - lag_index - subsequences_length >= 0
        # for all lag_index, hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(
                F.slice_axis(
                    sequence, axis=1, begin=begin_index, end=end_index
                )
            )

        return F.stack(*lagged_values, axis=-1)

    def create_network_input(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, num_features, history_length)
        past_target: Tensor,  # (batch_size, history_length, 1)
        past_observed_values: Tensor,  # (batch_size, history_length)
        future_time_feat: Optional[
            Tensor
        ],  # (batch_size, num_features, prediction_length)
        future_target: Optional[Tensor],  # (batch_size, prediction_length)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Creates inputs for the transformer network.
        All tensor arguments should have NTC layout.
        """

        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=None,
            )
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = F.concat(
                past_time_feat.slice_axis(
                    axis=1,
                    begin=self.history_length - self.context_length,
                    end=None,
                ),
                future_time_feat,
                dim=1,
            )
            sequence = F.concat(past_target, future_target, dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags = self.get_lagged_subsequences(
            F=F,
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = F.concat(
            embedded_cat,
            F.log(scale)
            if len(self.target_shape) == 0
            else F.log(scale.squeeze(axis=1)),
            dim=1,
        )

        repeated_static_feat = static_feat.expand_dims(axis=1).repeat(
            axis=1, repeats=subsequences_length
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = F.reshape(
            data=lags_scaled,
            shape=(
                -1,
                subsequences_length,
                len(self.lags_seq) * prod(self.target_shape),
            ),
        )

        # (batch_size, sub_seq_len, input_dim)
        inputs = F.concat(input_lags, time_feat, repeated_static_feat, dim=-1)

        return inputs, scale, static_feat

    @staticmethod
    def upper_triangular_mask(F, d):
        mask = F.zeros_like(F.eye(d))
        for k in range(d - 1):
            mask = mask + F.eye(d, d, k + 1)
        return mask * LARGE_NEGATIVE_VALUE


class TransformerTrainingNetwork(TransformerNetwork):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
    ) -> Tensor:
        """
        Computes the loss for training Transformer, all inputs tensors representing time series have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape, seq_len)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target : (batch_size, prediction_length, *target_shape)

        Returns
        -------
        Loss with shape (batch_size, context + prediction_length, 1)
        """

        # create the inputs for the encoder
        inputs, scale, _ = self.create_network_input(
            F=F,
            feat_static_cat=feat_static_cat,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        enc_input = F.slice_axis(
            inputs, axis=1, begin=0, end=self.context_length
        )
        dec_input = F.slice_axis(
            inputs, axis=1, begin=self.context_length, end=None
        )

        # pass through encoder
        enc_out = self.encoder(enc_input)

        # input to decoder
        dec_output = self.decoder(
            dec_input,
            enc_out,
            self.upper_triangular_mask(F, self.prediction_length),
        )

        # compute loss
        distr_args = self.proj_dist_args(dec_output)
        distr = self.distr_output.distribution(distr_args, scale=scale)
        loss = distr.loss(future_target)

        return loss.mean()


class TransformerPredictionNetwork(TransformerNetwork):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        F,
        static_feat: Tensor,
        past_target: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        enc_out: Tensor,
    ) -> Tensor:
        """
        Computes sample paths by unrolling the LSTM starting with a initial input and state.

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length, 1).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch. Shape: (batch_size, ).
        enc_out: Tensor
            output of the encoder. Shape: (batch_size, num_cells)

        Returns
        --------
        sample_paths : Tensor
            a tensor containing sampled paths. Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size * self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_time_feat = time_feat.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_static_feat = static_feat.repeat(
            repeats=self.num_parallel_samples, axis=0
        ).expand_dims(axis=1)
        repeated_enc_out = enc_out.repeat(
            repeats=self.num_parallel_samples, axis=0
        ).expand_dims(axis=1)
        repeated_scale = scale.repeat(
            repeats=self.num_parallel_samples, axis=0
        )

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and update the state
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                F=F,
                sequence=repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = F.broadcast_div(
                lags, repeated_scale.expand_dims(axis=-1)
            )

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = F.reshape(
                data=lags_scaled,
                shape=(-1, 1, prod(self.target_shape) * len(self.lags_seq)),
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            dec_input = F.concat(
                input_lags,
                repeated_time_feat.slice_axis(axis=1, begin=k, end=k + 1),
                repeated_static_feat,
                dim=-1,
            )

            dec_output = self.decoder(dec_input, repeated_enc_out, None, False)

            distr_args = self.proj_dist_args(dec_output)

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(
                distr_args, scale=repeated_scale
            )

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample()

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = F.concat(
                repeated_past_target, new_samples, dim=1
            )
            future_samples.append(new_samples)

        # reset cache of the decoder
        self.decoder.cache_reset()

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = F.concat(*future_samples, dim=1)

        # (batch_size, num_samples, *target_shape, prediction_length)
        return samples.reshape(
            shape=(
                (-1, self.num_parallel_samples)
                + self.target_shape
                + (self.prediction_length,)
            )
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
    ) -> Tensor:
        """
        Predicts samples, all tensors should have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns predicted samples
        -------

        """

        # create the inputs for the encoder
        inputs, scale, static_feat = self.create_network_input(
            F=F,
            feat_static_cat=feat_static_cat,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=None,
            future_target=None,
        )

        # pass through encoder
        enc_out = self.encoder(inputs)

        return self.sampling_decoder(
            F=F,
            past_target=past_target,
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            enc_out=enc_out,
        )
