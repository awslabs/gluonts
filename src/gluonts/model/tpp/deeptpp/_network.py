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
import numpy as np
from mxnet import nd

from gluonts.core.component import validated
from gluonts.model.tpp import distribution
from gluonts.model.tpp.distribution.base import TPPDistributionOutput
from gluonts.mx import Tensor
from gluonts.mx.distribution import CategoricalOutput


# noinspection PyAbstractClass
class DeepTPPNetworkBase(mx.gluon.HybridBlock):
    """
    Temporal point process model based on a recurrent neural network.

    Parameters
    ----------
    num_marks
        Number of discrete marks (correlated processes), that are available
        in the data.
    interval_length
        The length of the total time interval that is in the prediction
        range. Note that in contrast to discrete-time models in the rest
        of GluonTS, the network is trained to predict an interval, in
        continuous time.
    time_distr_output
        Output distribution for the inter-arrival times. Available distributions
        can be found in gluonts.model.tpp.distribution.
    embedding_dim
        Dimension of vector embeddings of marks (used only as input).
    num_hidden_dimensions
        Number of hidden units in the RNN.
    output_scale
        Positive scaling applied to the inter-event times. You should provide
        this argument if the average inter-arrival time is much larger than 1.
    apply_log_to_rnn_inputs
        Apply logarithm to inter-event times that are fed into the RNN.
    """

    @validated()
    def __init__(
        self,
        num_marks: int,
        interval_length: float,
        time_distr_output: TPPDistributionOutput = distribution.WeibullOutput(),
        embedding_dim: int = 5,
        num_hidden_dimensions: int = 10,
        output_scale: Optional[Tensor] = None,
        apply_log_to_rnn_inputs: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_marks = num_marks
        self.interval_length = interval_length
        self.rnn_hidden_size = num_hidden_dimensions
        self.output_scale = output_scale
        self.apply_log_to_rnn_inputs = apply_log_to_rnn_inputs

        with self.name_scope():
            self.embedding = mx.gluon.nn.Embedding(
                input_dim=num_marks, output_dim=embedding_dim
            )
            self.rnn = mx.gluon.rnn.GRU(
                num_hidden_dimensions,
                input_size=embedding_dim + 1,
                layout="NTC",
            )
            # Conditional distribution over the inter-arrival times
            self.time_distr_output = time_distr_output
            self.time_distr_args_proj = self.time_distr_output.get_args_proj()
            # Conditional distribution over the marks
            if num_marks > 1:
                self.mark_distr_output = CategoricalOutput(num_marks)
                self.mark_distr_args_proj = (
                    self.mark_distr_output.get_args_proj()
                )

    def hybridize(self, active=True, **kwargs):
        if active:
            raise NotImplementedError(
                "DeepTPP blocks do not support hybridization"
            )


class DeepTPPTrainingNetwork(DeepTPPNetworkBase):

    # noinspection PyMethodOverriding,PyPep8Naming,PyIncorrectDocstring
    def hybrid_forward(
        self,
        F,
        target: Tensor,
        valid_length: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Computes the negative log likelihood loss for the given sequences.

        As the model is trained on past (resp. future) or context
        (resp. prediction) "intervals" as opposed to fixed-length "sequences",
        the number of data points available varies across observations. To
        account for this, data is made available to the training network as a
        "ragged" tensor. The number of valid entries in each sequence is
        provided in a separate variable, :code:`xxx_valid_length`.

        Parameters
        ----------
        F
            MXNet backend.
        target
            Tensor with observations.
            Shape: (batch_size, past_max_sequence_length, target_dim).
        valid_length
            The `valid_length` or number of valid entries in the past_target
            Tensor. Shape: (batch_size,)

        Returns
        -------
        Tensor
            Loss tensor. Shape: (batch_size,).
        """
        if F is mx.sym:
            raise ValueError(
                "The DeepTPP model currently doesn't support hybridization."
            )

        batch_size = target.shape[0]
        # IMPORTANT: We add an additional zero at the end of each sequence
        # It will be used to store the time until the end of the interval
        target = F.concat(target, F.zeros((batch_size, 1, 2)), dim=1)
        # (N, T + 1, 2)

        ia_times, marks = F.split(
            target, num_outputs=2, axis=-1
        )  # inter-arrival times, marks
        marks = marks.squeeze(axis=-1)  # (N, T + 1)

        valid_length = valid_length.reshape(-1).astype(
            ia_times.dtype
        )  # make sure shape is (batch_size,)

        if self.apply_log_to_rnn_inputs:
            ia_times_input = ia_times.clip(1e-8, np.inf).log()
        else:
            ia_times_input = ia_times
        rnn_input = F.concat(ia_times_input, self.embedding(marks), dim=-1)
        rnn_output = self.rnn(rnn_input)  # (N, T + 1, H)

        rnn_init_state = F.zeros([batch_size, 1, self.rnn_hidden_size])
        history_emb = F.slice_axis(
            F.concat(rnn_init_state, rnn_output, dim=1),
            axis=1,
            begin=0,
            end=-1,
        )  # (N, T + 1, H)

        # Augment ia_times by adding the time remaining until interval_length
        # Afterwards, each row of ia_times will sum up to interval_length
        ia_times = ia_times.squeeze(axis=-1)  # (N, T + 1)
        time_remaining = self.interval_length - ia_times.sum(-1)  # (N)
        # Equivalent to ia_times[F.arange(N), valid_length] = time_remaining
        indices = F.stack(F.arange(batch_size), valid_length)
        time_remaining_tensor = F.scatter_nd(
            time_remaining, indices, ia_times.shape
        )
        ia_times_aug = ia_times + time_remaining_tensor

        time_distr_args = self.time_distr_args_proj(history_emb)
        time_distr = self.time_distr_output.distribution(
            time_distr_args, scale=self.output_scale
        )
        log_intensity = time_distr.log_intensity(ia_times_aug)  # (N, T + 1)
        log_survival = time_distr.log_survival(ia_times_aug)  # (N, T + 1)

        if self.num_marks > 1:
            mark_distr_args = self.mark_distr_args_proj(history_emb)
            mark_distr = self.mark_distr_output.distribution(mark_distr_args)
            log_intensity = log_intensity + mark_distr.log_prob(marks)

        def _mask(x, sequence_length):
            return F.SequenceMask(
                data=x,
                sequence_length=sequence_length,
                axis=1,
                use_sequence_length=True,
            )

        log_likelihood = F.sum(
            (
                _mask(log_intensity, valid_length)
                + _mask(log_survival, valid_length + 1)
            ),
            axis=-1,
        )  # (N)

        return -log_likelihood


class DeepTPPPredictionNetwork(DeepTPPNetworkBase):
    @validated()
    def __init__(
        self,
        prediction_interval_length: float,
        num_parallel_samples: int = 100,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.prediction_interval_length = prediction_interval_length

    # noinspection PyMethodOverriding,PyPep8Naming,PyIncorrectDocstring
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_valid_length: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Draw forward samples from the model. At each step, we sample an
        inter-event time and feed it into the RNN to obtain the parameters for
        the next distribution over the inter-event time.

        Parameters
        ----------
        F
            MXNet backend.
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim). Has to comply
            with :code:`self.context_interval_length`.
        past_valid_length
            The `valid_length` or number of valid entries in the past_target
            Tensor. Shape: (batch_size,)

        Returns
        -------
        sampled_target: Tensor
            Predicted inter-event times and marks.
            Shape: (samples, batch_size, max_prediction_length, target_dim).
        sampled_valid_length: Tensor
            The number of valid entries in the time axis of each sample.
            Shape (samples, batch_size)
        """
        # Variable-length generation (while t < t_max) is a potential problem
        if F is mx.sym:
            raise ValueError(
                "The DeepTPP model currently doesn't support hybridization."
            )

        assert (
            past_target.shape[-1] == 2
        ), "TPP data should have two target_dim, interarrival times and marks"

        batch_size = past_target.shape[0]

        # condition the prediction network on the past events
        past_ia_times, past_marks = F.split(
            past_target, num_outputs=2, axis=-1
        )
        past_valid_length = past_valid_length.reshape(-1).astype(
            past_ia_times.dtype
        )

        if self.apply_log_to_rnn_inputs:
            past_ia_times_input = past_ia_times.clip(1e-8, np.inf).log()
        else:
            past_ia_times_input = past_ia_times
        rnn_input = F.concat(
            past_ia_times_input,
            self.embedding(past_marks.squeeze(axis=-1)),
            dim=-1,
        )
        rnn_output = self.rnn(rnn_input)  # (N, T, H)
        rnn_init_state = F.zeros([batch_size, 1, self.rnn_hidden_size])

        past_history_emb = F.concat(
            rnn_init_state, rnn_output, dim=1
        )  # (N, T + 1, H)

        # Select the history embedding after the last event in the past
        indices = F.stack(F.arange(batch_size), past_valid_length)
        history_emb = F.gather_nd(past_history_emb, indices)  # (N, H)

        num_total_samples = self.num_parallel_samples * batch_size
        history_emb = history_emb.expand_dims(0).repeat(
            self.num_parallel_samples, axis=0
        )  # (S, N, H)
        history_emb = history_emb.reshape(
            [num_total_samples, self.rnn_hidden_size]
        )  # (S * N, H)

        sampled_ia_times_list: List[nd.NDArray] = []
        sampled_marks_list: List[nd.NDArray] = []
        arrival_times = F.zeros([num_total_samples])

        # Time from the last observed event until the past interval end
        past_time_elapsed = past_ia_times.squeeze(axis=-1).sum(-1)
        past_time_remaining = self.interval_length - past_time_elapsed  # (N)
        past_time_remaining_repeat = (
            past_time_remaining.expand_dims(0)
            .repeat(self.num_parallel_samples, axis=0)
            .reshape([num_total_samples])
        )  # (S * N)

        first_step = True
        while F.sum(arrival_times < self.prediction_interval_length) > 0:
            # Sample the next inter-arrival time
            time_distr_args = self.time_distr_args_proj(history_emb)
            time_distr = self.time_distr_output.distribution(
                time_distr_args,
                scale=self.output_scale,
            )
            if first_step:
                # Time from the last event until the next event
                next_ia_times = time_distr.sample(
                    lower_bound=past_time_remaining_repeat
                )
                # Time from the prediction interval start until the next event
                clipped_ia_times = next_ia_times - past_time_remaining_repeat
                sampled_ia_times_list.append(clipped_ia_times)
                arrival_times = arrival_times + clipped_ia_times
                first_step = False
            else:
                next_ia_times = time_distr.sample()
                sampled_ia_times_list.append(next_ia_times)
                arrival_times = arrival_times + next_ia_times

            # Sample the next marks
            if self.num_marks > 1:
                mark_distr_args = self.mark_distr_args_proj(history_emb)
                next_marks = self.mark_distr_output.distribution(
                    mark_distr_args
                ).sample()
            else:
                next_marks = F.zeros([num_total_samples])

            sampled_marks_list.append(next_marks)

            # Pass the generated ia_times & marks into the RNN to obtain
            # the next history embedding
            if self.apply_log_to_rnn_inputs:
                next_ia_times_input = next_ia_times.clip(1e-8, np.inf).log()
            else:
                next_ia_times_input = next_ia_times
            rnn_input = F.concat(
                next_ia_times_input.expand_dims(-1),
                self.embedding(next_marks),
                dim=-1,
            ).expand_dims(1)

            history_emb = self.rnn(rnn_input).squeeze(axis=1)  # (S * N, C)

        sampled_ia_times = F.stack(*sampled_ia_times_list, axis=-1)
        sampled_marks = F.stack(*sampled_marks_list, axis=-1).astype("float32")

        sampled_valid_length = F.sum(
            F.cumsum(sampled_ia_times, axis=1)
            < self.prediction_interval_length,
            axis=-1,
        )

        def _mask(x, sequence_length):
            return F.SequenceMask(
                data=x,
                sequence_length=sequence_length,
                axis=1,
                use_sequence_length=True,
            )

        sampled_ia_times = _mask(sampled_ia_times, sampled_valid_length)
        sampled_marks = _mask(sampled_marks, sampled_valid_length)

        sampled_ia_times = sampled_ia_times.reshape(
            [self.num_parallel_samples, batch_size, -1]
        )
        sampled_marks = sampled_marks.reshape(
            [self.num_parallel_samples, batch_size, -1]
        )
        sampled_valid_length = sampled_valid_length.reshape(
            [self.num_parallel_samples, batch_size]
        )
        sampled_target = F.stack(sampled_ia_times, sampled_marks, axis=-1)
        return sampled_target, sampled_valid_length
