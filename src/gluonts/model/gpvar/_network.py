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
from gluonts.model.deepvar._network import DeepVARNetwork
from gluonts.mx import Tensor
from gluonts.mx.distribution.distribution import getF
from gluonts.mx.distribution.lowrank_gp import LowrankGPOutput


class GPVARNetwork(DeepVARNetwork):
    @validated()
    def __init__(self, target_dim_sample: int, **kwargs) -> None:
        super().__init__(embedding_dimension=1, cardinality=[1], **kwargs)
        self.target_dim_sample = target_dim_sample

        assert isinstance(self.distr_output, LowrankGPOutput)

        with self.name_scope():
            self.embed = mx.gluon.nn.Embedding(
                input_dim=self.target_dim,
                output_dim=4 * self.distr_output.rank,
            )

    def unroll(
        self,
        F,
        lags: Tensor,
        scale: Tensor,
        time_feat: Tensor,
        target_dimension_indicator: Tensor,
        unroll_length: int,
        begin_state: Optional[List[Tensor]],
    ) -> Tuple[Tensor, List[Tensor], Tensor, Tensor]:
        """
        Prepares the input to the RNN and unrolls it the given number of time
        steps.

        Parameters
        ----------
        F
        lags
            Input lags (batch_size, sub_seq_len, target_dim, num_lags)
        scale
            Mean scale (batch_size, 1, target_dim)
        time_feat
            Additional time features
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        unroll_length
            length to unroll
        begin_state
            State to start the unrolling of the RNN

        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN
        """
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

        outputs = []
        states = []

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimension_indicator)

        # (batch_size, seq_len, target_dim, embed_dim)
        repeated_index_embeddings = index_embeddings.expand_dims(
            axis=1
        ).repeat(axis=1, repeats=unroll_length)

        inputs_seq = []

        for i in range(self.target_dim_sample):
            # (batch_size, sub_seq_len, input_dim)
            inputs = F.concat(
                lags_scaled.slice_axis(axis=2, begin=i, end=i + 1).squeeze(
                    axis=2
                ),
                repeated_index_embeddings.slice_axis(
                    axis=2, begin=i, end=i + 1
                ).squeeze(axis=2),
                # all_input_lags,
                time_feat,
                dim=-1,
            )

            # unroll encoder
            # outputs: (batch_size, sub_seq_len, num_hidden)
            # state: tuple of (batch_size, num_hidden)
            outputs_single_dim, state = self.rnn.unroll(
                inputs=inputs,
                length=unroll_length,
                layout="NTC",
                merge_outputs=True,
                begin_state=begin_state[i]
                if begin_state is not None
                else None,
            )
            outputs.append(outputs_single_dim)
            states.append(state)
            inputs_seq.append(inputs)

        # (batch_size, seq_len, target_dim, num_cells)
        outputs = F.stack(*outputs, num_args=self.target_dim_sample, axis=2)

        return outputs, states, lags_scaled, time_feat

    def distr(
        self,
        rnn_outputs: Tensor,
        time_features: Tensor,
        scale: Tensor,
        lags_scaled: Tensor,
        target_dimension_indicator: Tensor,
        seq_len: int,
    ):
        """
        Returns the distribution of GPVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        time_features
            Dynamic time features (batch_size, seq_len, num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags used for RNN input
            (batch_size, seq_len, target_dim, num_lags)
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        seq_len
            Length of the sequences

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        F = getF(rnn_outputs)

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimension_indicator)

        # broadcast to (batch_size, seq_len, target_dim, embed_dim)
        repeated_index_embeddings = index_embeddings.expand_dims(
            axis=1
        ).repeat(axis=1, repeats=seq_len)

        # broadcast to (batch_size, seq_len, target_dim, num_features)
        time_features = time_features.expand_dims(axis=2).repeat(
            axis=2, repeats=self.target_dim_sample
        )

        # (batch_size, seq_len, target_dim, embed_dim + num_cells + num_inputs)
        distr_input = F.concat(
            rnn_outputs, repeated_index_embeddings, time_features, dim=-1
        )

        # TODO 1 pass inputs in proj args
        distr_args = self.proj_dist_args(distr_input)

        # compute likelihood of target given the predicted parameters
        assert isinstance(self.distr_output, LowrankGPOutput)
        distr = self.distr_output.distribution(
            distr_args, scale=scale, dim=self.target_dim_sample
        )

        return distr, distr_args


class GPVARTrainingNetwork(GPVARNetwork):

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        target_dimension_indicator: Tensor,
        past_time_feat: Tensor,
        past_target_cdf: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
        future_target_cdf: Tensor,
        future_observed_values: Tensor,
    ) -> Tuple[Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        F
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
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

        return self.train_hybrid_forward(
            F,
            target_dimension_indicator,
            past_time_feat,
            past_target_cdf,
            past_observed_values,
            past_is_pad,
            future_time_feat,
            future_target_cdf,
            future_observed_values,
        )


class GPVARPredictionNetwork(GPVARNetwork):
    @validated()
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to the
        # last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def make_states(self, begin_states: List[Tensor]) -> List[List[Tensor]]:
        """
        Repeat states to match the the shape induced by the number of sample
        paths.

        Parameters
        ----------
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)

        Returns
        -------
            List of list of initial states
        """

        def repeat(tensor):
            return tensor.repeat(repeats=self.num_parallel_samples, axis=0)

        return [[repeat(s) for s in states] for states in begin_states]

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        target_dimension_indicator: Tensor,
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target_cdf: Tensor,  # (batch_size, history_length, target_dim)
        past_observed_values: Tensor,  # (batch_size, history_length, target_dim)
        past_is_pad: Tensor,
        future_time_feat: Tensor,  # (batch_size, prediction_length, num_features)
    ) -> Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------
        F
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
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

        """

        return self.predict_hybrid_forward(
            F=F,
            target_dimension_indicator=target_dimension_indicator,
            past_time_feat=past_time_feat,  # (batch_size, history_length, num_features)
            past_target_cdf=past_target_cdf,  # (batch_size, history_length, target_dim)
            past_observed_values=past_observed_values,  # (batch_size, history_length, target_dim)
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,  # (batch_size, prediction_length, num_features)
        )
