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

# Standard library imports
from typing import Optional
import warnings

# Third-party imports
import mxnet as mx
from mxnet.gluon import nn, loss

# First-party imports
from gluonts.model.common import Tensor
from gluonts.core.component import validated, DType
from gluonts.block.scaler import MeanScaler, NOPScaler


class LSTNetBase(nn.HybridBlock):
    @validated()
    def __init__(
        self,
        num_series: int,
        channels: int,
        kernel_size: int,
        rnn_cell_type: str,
        rnn_num_layers: int,
        skip_rnn_cell_type: str,
        skip_rnn_num_layers: int,
        skip_size: int,
        ar_window: int,
        context_length: int,
        horizon: Optional[int],
        prediction_length: Optional[int],
        dropout_rate: float,
        output_activation: Optional[str],
        scaling: bool,
        dtype: DType,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_series = num_series
        self.channels = channels
        assert (
            channels % skip_size == 0
        ), "number of conv1d `channels` must be divisible by the `skip_size`"
        self.skip_size = skip_size
        assert (
            ar_window > 0
        ), "auto-regressive window must be a positive integer"
        self.ar_window = ar_window
        assert not ((horizon is None)) == (
            prediction_length is None
        ), "Exactly one of `horizon` and `prediction_length` must be set at a time"
        assert (
            horizon is None or horizon > 0
        ), "`horizon` must be greater than zero"
        assert (
            prediction_length is None or prediction_length > 0
        ), "`prediction_length` must be greater than zero"
        self.prediction_length = prediction_length
        self.horizon = horizon
        assert context_length > 0, "`context_length` must be greater than zero"
        self.context_length = context_length
        if output_activation is not None:
            assert output_activation in [
                "sigmoid",
                "tanh",
            ], "`output_activation` must be either 'sigmiod' or 'tanh' "
        self.output_activation = output_activation
        assert rnn_cell_type in [
            "gru",
            "lstm",
        ], "`rnn_cell_type` must be either 'gru' or 'lstm' "
        assert skip_rnn_cell_type in [
            "gru",
            "lstm",
        ], "`skip_rnn_cell_type` must be either 'gru' or 'lstm' "
        self.conv_out = context_length - kernel_size + 1
        conv_skip = self.conv_out // skip_size
        assert conv_skip > 0, (
            "conv1d output size must be greater than or equal to `skip_size`\n"
            "Choose a smaller `kernel_size` or bigger `context_length`"
        )
        self.channel_skip_count = conv_skip * skip_size
        self.skip_rnn_c_dim = channels * skip_size
        self.dtype = dtype
        with self.name_scope():
            self.cnn = nn.Conv1D(
                channels,
                kernel_size,
                activation="relu",
                layout="NCW",
                in_channels=num_series,
            )  # NCT
            self.cnn.cast(dtype)
            self.dropout = nn.Dropout(dropout_rate)
            self.rnn = self._create_rnn_layer(
                channels, rnn_num_layers, rnn_cell_type, dropout_rate
            )  # NTC
            self.rnn.cast(dtype)
            self.skip_rnn = self._create_rnn_layer(
                channels, skip_rnn_num_layers, skip_rnn_cell_type, dropout_rate
            )  # NTC
            self.skip_rnn.cast(dtype)
            # TODO: add temporal attention option
            self.fc = nn.Dense(num_series, dtype=dtype)
            if self.horizon:
                self.ar_fc = nn.Dense(1, dtype=dtype)
            else:
                self.ar_fc = nn.Dense(prediction_length, dtype=dtype)
            if scaling:
                self.scaler = MeanScaler()
            else:
                self.scaler = NOPScaler()

    @staticmethod
    def _create_rnn_layer(
        num_cells: int, num_layers: int, cell_type: str, dropout_rate: float
    ) -> nn.HybridBlock:
        # TODO: GRUCell activation is fixed to tanh
        RnnCell = {"lstm": mx.gluon.rnn.LSTMCell, "gru": mx.gluon.rnn.GRUCell}[
            cell_type
        ]
        rnn = mx.gluon.rnn.HybridSequentialRNNCell()
        for _ in range(num_layers):
            cell = RnnCell(hidden_size=num_cells)
            cell = (
                mx.gluon.rnn.ZoneoutCell(cell, zoneout_states=dropout_rate)
                if dropout_rate > 0.0
                else cell
            )
            rnn.add(cell)
        return rnn

    def _skip_rnn_layer(self, F, x: Tensor) -> Tensor:
        skip_c = F.slice_axis(
            x, axis=1, begin=-self.channel_skip_count, end=None  # NCT
        )
        skip_c = F.reshape(
            skip_c, shape=(0, 0, -1, self.skip_size)
        )  # NTCxskip
        skip_c = F.transpose(skip_c, axes=(0, 3, 1, 2))  # NxskipxTxC
        skip_c = F.reshape(skip_c, shape=(-3, 0, -1))  # (Nxskip)TC
        if F is mx.ndarray:
            ctx = (
                skip_c.context
                if isinstance(skip_c, mx.gluon.tensor_types)
                else skip_c[0].context
            )
            with ctx:
                begin_state = self.skip_rnn.begin_state(
                    func=F.zeros, dtype=self.dtype, batch_size=skip_c.shape[0]
                )
        else:
            begin_state = self.skip_rnn.begin_state(
                func=F.zeros, dtype=self.dtype, batch_size=0
            )

        s, _ = self.skip_rnn.unroll(
            inputs=skip_c,
            length=min(self.channel_skip_count, self.context_length),
            layout="NTC",
            merge_outputs=True,
            begin_state=begin_state,
        )
        s = F.squeeze(
            F.slice_axis(s, axis=1, begin=-1, end=None), axis=1
        )  # (Nxskip)xC
        s = F.reshape(s, shape=(-1, self.skip_rnn_c_dim))  # Nx(skipxC)
        return s

    def _ar_highway(self, F, x: Tensor) -> Tensor:
        ar_x = F.slice_axis(x, axis=2, begin=-self.ar_window, end=None)  # NCT
        ar_x = F.reshape(ar_x, shape=(-3, 0))  # (NC)xT
        ar = self.ar_fc(ar_x)  # (NC)x(1 or prediction_length)
        if self.horizon:
            ar = F.reshape(ar, shape=(-1, self.num_series, 1))
        else:
            ar = F.reshape(
                ar, shape=(-1, self.num_series, self.prediction_length)
            )
        return ar

    def hybrid_forward(
        self, F, past_target: Tensor, past_observed_values: Tensor
    ) -> Tensor:
        """
        Given the tensor `past_target`, first we normalize it by the
        `past_observed_values` which is an indicator tensor with 0 or 1 values.
        Then it outputs the result of LSTNet.

        Parameters
        ----------
        F
        past_target
            Tensor of shape (batch_size, num_series, context_length)
        past_observed_values
            Tensor of shape (batch_size, num_series, context_length)

        Returns
        -------
        Tensor
            Shape (batch_size, num_series, 1) if `horizon` was specified
            and of shape (batch_size, num_series, prediction_length)
            if `prediction_length` was provided
            
        """

        scaled_past_target, _ = self.scaler(
            past_target.slice_axis(
                axis=2, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=2, begin=-self.context_length, end=None
            ),
        )
        c = self.cnn(scaled_past_target)
        c = self.dropout(c)
        c = F.transpose(c, axes=(0, 2, 1))  # NTC

        if F is mx.ndarray:
            ctx = (
                c.context
                if isinstance(c, mx.gluon.tensor_types)
                else c[0].context
            )
            with ctx:
                rnn_begin_state = self.rnn.begin_state(
                    func=F.zeros, dtype=self.dtype, batch_size=c.shape[0]
                )
        else:
            rnn_begin_state = self.rnn.begin_state(
                func=F.zeros, dtype=self.dtype, batch_size=0
            )
        r, _ = self.rnn.unroll(
            inputs=c,
            length=min(self.conv_out, self.context_length),
            layout="NTC",
            merge_outputs=True,
            begin_state=rnn_begin_state,
        )
        r = F.squeeze(
            F.slice_axis(r, axis=1, begin=-1, end=None), axis=1
        )  # NC
        s = self._skip_rnn_layer(F, c)
        # make fc broadcastable for output
        fc = self.fc(F.concat(r, s, dim=1)).expand_dims(
            axis=2
        )  # N x num_series x 1
        if self.prediction_length:
            fc = F.tile(
                fc, reps=(1, 1, self.prediction_length)
            )  # N x num_series x prediction_length
        ar = self._ar_highway(F, past_target)
        out = fc + ar
        if self.output_activation is None:
            return out
        return (
            F.sigmoid(out)
            if self.output_activation == "sigmoid"
            else F.tanh(out)
        )


class LSTNetTrain(LSTNetBase):
    @validated()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = loss.L1Loss()

    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_target: Tensor,
    ) -> Tensor:
        """
        Computes the training l1 loss for LSTNet for multivariate time-series.
        All input tensors have NCT layout.

        Parameters
        ----------
        F
        past_target
            Tensor of shape (batch_size, num_series, context_length)
        past_observed_values
            Tensor of shape (batch_size, num_series, context_length)
        future_target
            Tensor of shape (batch_size, num_series, 1) if `horizon` was specified
            and of shape (batch_size, num_series, prediction_length)
            if `prediction_length` was provided

        Returns
        -------
        Tensor
            Loss values of shape (batch_size,)
        """

        ret = super().hybrid_forward(F, past_target, past_observed_values)
        if self.horizon:
            # get the last time horizon
            future_target = F.slice_axis(
                future_target, axis=2, begin=-1, end=None
            )
        loss = self.loss_fn(ret, future_target)
        return loss


class LSTNetPredict(LSTNetBase):
    def hybrid_forward(
        self, F, past_target: Tensor, past_observed_values: Tensor
    ) -> Tensor:
        """
        Returns the LSTNet predictions. All tensors have NCT layout.

        Parameters
        ----------
        F
        past_target
            Tensor of shape (batch_size, num_series, context_length)
        past_observed_values
            Tensor of shape (batch_size, num_series, context_length)

        Returns
        -------
        Tensor
            Predicted samples of shape (num_samples, 1, num_series) when using `horizon`
            and of shape (num_samples, prediction_length, num_series)
            when providing `prediction_length`
        """

        ret = super().hybrid_forward(F, past_target, past_observed_values)
        ret = F.swapaxes(ret, 1, 2)
        return ret.expand_dims(axis=1)
