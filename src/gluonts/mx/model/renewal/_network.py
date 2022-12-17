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

import mxnet as mx
from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.distribution import Distribution, DistributionOutput
from gluonts.mx.distribution.distribution import getF
from mxnet import gluon


class DeepRenewalNetwork(gluon.HybridBlock):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        interval_distr_output: DistributionOutput,
        size_distr_output: DistributionOutput,
        num_cells: int,
        num_layers: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.interval_distr_output = interval_distr_output
        self.size_distr_output = size_distr_output
        self.num_cells = num_cells
        self.num_layers = num_layers

        with self.name_scope():
            self.rnn_cell = mx.gluon.rnn.LSTM(
                hidden_size=num_cells,
                num_layers=num_layers,
                layout="NTC",
                dropout=dropout_rate,
                h2h_weight_initializer=mx.init.Xavier(),
                h2r_weight_initializer=mx.init.Xavier(),
            )
            self.dense = mx.gluon.nn.Dense(
                units=2,
                activation="softrelu",
                flatten=False,
                weight_initializer=mx.init.Xavier(),
                bias_initializer=mx.init.Constant(5.0),
            )

        # add constant learnable parameters if the distributions require
        # more than the conditional mean
        with self.name_scope():
            if len(self.interval_distr_output.args_dim) > 1:
                self.interval_alpha_bias = self.params.get(
                    "interval_alpha_bias",
                    shape=(1,),
                    init=mx.init.Constant(2),
                )
            if len(self.size_distr_output.args_dim) > 1:
                self.size_alpha_bias = self.params.get(
                    "size_alpha_bias",
                    shape=(1,),
                    init=mx.init.Constant(2),
                )

    def begin_state(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return self.rnn_cell.begin_state(batch_size=batch_size)

    def distribution(
        self,
        cond_mean: Tensor,
        interval_alpha_bias: Optional[Tensor] = None,
        size_alpha_bias: Optional[Tensor] = None,
    ) -> Tuple[Distribution, ...]:
        F = getF(cond_mean)

        cond_interval, cond_size = F.split(cond_mean, num_outputs=2, axis=-1)

        alpha_biases = [
            F.broadcast_mul(F.ones_like(cond_interval), bias)
            if bias is not None
            else None
            for bias in [interval_alpha_bias, size_alpha_bias]
        ]

        distr_params = zip(
            [self.interval_distr_output, self.size_distr_output],
            [cond_interval, cond_size],
            alpha_biases,
        )

        return tuple(
            (
                do.distribution(mean)
                if len(do.args_dim) == 1
                else do.distribution(
                    [mean, F.Activation(alpha_bias, "softrelu") + 1e-5]
                )
            )
            for ix, (do, mean, alpha_bias) in enumerate(distr_params)
        )

    @staticmethod
    def forwardshift(A):
        """
        Shift an array's content forward by 1 time step along the first axis,
        keeping the shape identical by padding on the left with zeros.

        Parameters
        ----------
        A : nd.NDArray
            Shape (N, T, ...), the tensor in which the entries will be shifted
            forward by one
        """
        F = getF(A)
        A = F.Concat(
            F.zeros_like(F.slice_axis(A, axis=1, begin=0, end=1)), A, dim=1
        )
        return F.slice_axis(A, axis=1, begin=0, end=-1)

    def mu_map(
        self,
        data: Tensor,
        shift: bool = True,
        state=None,
    ) -> Tensor:
        """
        Map a given (N, T, 2) tensor to conditional interval and size means of
        the next time step.

        Parameters
        ----------
        data: Tensor
            tensor of shape (N, T, 2) containing the past target, with the
            first array along the last dimension containing intervals, and
            the second array containing demand sizes.
        shift: bool
            if True, the past data will be shifted forward by one time step
        state
            RNN cell state if available

        Returns
        -------
        conditional_means: Tensor
            tensor of shape (N, T, 2) containing conditional means for
            intervals and sizes respectively
        rnn_out_state
            output state of the RNN, returned only if the input `state` is not
            None
        """
        data = self.forwardshift(data) if shift else data

        if state is not None:
            rnn_out, rnn_out_state = self.rnn_cell(data, state)
        else:
            rnn_out = self.rnn_cell(data)

        dense_out = self.dense(rnn_out)
        model_mu = dense_out + 1e-5

        return model_mu if state is None else (model_mu, rnn_out_state)


class DeepRenewalTrainingNetwork(DeepRenewalNetwork):
    def hybrid_forward(
        self,
        F,
        past_target,
        valid_length,
        interval_alpha_bias=None,
        size_alpha_bias=None,
        **kwargs,
    ) -> Tensor:
        """
        Compute negative log likelihood losses for given data.

        Parameters
        ----------
        F
        past_target
            shape (batch_size, history_length, 2). Expects a right-aligned
            ragged tensor containing the past data in an interval-size
            format.
        valid_length
            number of valid data points in each array of the batch. i.e.,
            non-valid points will be masked out. shape (batch_size, 1)
        interval_alpha_bias
        size_alpha_bias

        Returns
        -------
        loss
            negative log likelihood with shape (batch_size, history_length)
        """
        cond_mean = self.mu_map(past_target, shift=True)
        dist_interval, dist_size = self.distribution(
            cond_mean, interval_alpha_bias, size_alpha_bias
        )
        data_interval, data_size = F.split(past_target, num_outputs=2, axis=-1)

        # TODO: on windows, operators below may produce NaN values

        log_prob = F.squeeze(
            dist_interval.log_prob(data_interval - 1)
            + dist_size.log_prob(data_size - 1),
            axis=-1,
        )
        reverse_lengths = F.broadcast_sub(
            F.maximum(1, F.max(valid_length)), valid_length
        ).squeeze(-1)
        mask = F.SequenceMask(
            F.ones_like(log_prob),
            reverse_lengths,
            use_sequence_length=True,
            axis=1,
        )
        loss = -F.where(1 - mask, log_prob, F.zeros_like(log_prob))

        return loss


class DeepRenewalPredictionNetwork(DeepRenewalNetwork):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

    def sampling_decoder(
        self,
        F,
        past_target: Tensor,
        interval_alpha_bias: Optional[Tensor] = None,
        size_alpha_bias: Optional[Tensor] = None,
        time_remaining: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample a trajectory of interval-size pairs from the model given past
        target.

        Parameters
        ----------
        past_target
            (batch_size, history_length, 2) shaped tensor containing the past
            time series in an interval-size representation.
        time_remaining
            (batch_size, 1) shaped tensor containing the number of time steps
            that were zero before the start of the forecast horizon

        Returns
        -------
        samples
            samples of shape (batch_size, num_parallel_samples, 2,
            sequence_length)
        """
        if time_remaining is None:
            time_remaining = F.broadcast_like(
                F.zeros((1,)), past_target, lhs_axes=(0,), rhs_axes=(0,)
            )

        # if no valid points in the past_target, override time remaining with
        # zero
        batch_sum = F.sum(past_target, axis=1).sum(-1).expand_dims(-1)
        time_remaining = F.where(
            batch_sum > 0, time_remaining, F.zeros_like(time_remaining)
        )

        repeated_past_target = past_target.repeat(
            repeats=self.num_parallel_samples, axis=0
        )  # (N * samples, T, 2)
        repeated_time_remaining = time_remaining.repeat(
            repeats=self.num_parallel_samples, axis=0
        )  # (N * samples, 1)

        interval_samples = []
        size_samples = []

        for t in range(self.prediction_length):
            cond_mean = self.mu_map(repeated_past_target, shift=False)
            cond_mean_last = F.slice_axis(
                cond_mean, axis=1, begin=-1, end=None
            ).squeeze(1)

            dist_interval, dist_size = self.distribution(
                cond_mean_last, interval_alpha_bias, size_alpha_bias
            )

            # initial samples for interval should be taken conditionally
            # we achieve this via (leaky) rejection sampling
            if t == 0:
                interval_sample = F.zeros_like(repeated_time_remaining)
                for j in range(50):
                    interval_sample = F.where(
                        interval_sample > repeated_time_remaining,
                        interval_sample,
                        dist_interval.sample(),
                    )
                interval_sample = (
                    F.where(
                        interval_sample <= repeated_time_remaining,
                        repeated_time_remaining,
                        interval_sample,
                    )
                    + 1
                )
            else:
                interval_sample = dist_interval.sample() + 1
            size_sample = dist_size.sample() + 1

            interval_samples.append(interval_sample)
            size_samples.append(size_sample)

            repeated_past_target = F.concat(
                repeated_past_target,
                F.concat(interval_sample, size_sample, dim=-1).expand_dims(1),
                dim=1,
            )

        interval_samples[0] = interval_samples[0] - repeated_time_remaining
        samples = F.concat(
            *[
                F.concat(x, y, dim=-1).expand_dims(1)
                for x, y in zip(interval_samples, size_samples)
            ],
            dim=1,
        )

        return samples.reshape(
            shape=(-1, self.num_parallel_samples) + samples.shape[1:]
        ).swapaxes(2, 3)

    def hybrid_forward(
        self,
        F,
        past_target,
        time_remaining,
        interval_alpha_bias=None,
        size_alpha_bias=None,
        **kwargs,
    ) -> Tensor:
        return self.sampling_decoder(
            F,
            past_target=past_target,
            interval_alpha_bias=interval_alpha_bias,
            size_alpha_bias=size_alpha_bias,
            time_remaining=time_remaining,
        )
