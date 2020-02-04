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
        prediction_length: int,
        dropout_rate: float,
        output_activation: Optional[str],
        scaling: bool,
        dtype: DType,
        *args,
        **kwargs
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
        assert (
            prediction_length > 0
        ), "`prediction_length` must be greater than zero"
        self.prediction_length = prediction_length
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
            "conv1d output size must be greater than or equal to skip_size\n"
            "Choose a smaller kernel_size or bigger context_length"
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
            self.ar_fc = nn.Dense(1, dtype=dtype)
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
        s, _ = self.skip_rnn.unroll(
            inputs=skip_c,
            length=min(self.channel_skip_count, self.context_length),
            layout="NTC",
            merge_outputs=True,
            begin_state=self.skip_rnn.begin_state(
                func=F.zeros,
                dtype=self.dtype,
                batch_size=skip_c.shape[0]
                if isinstance(skip_c, mx.nd.NDArray)
                else 0,
            ),
        )
        s = F.squeeze(
            F.slice_axis(s, axis=1, begin=-1, end=None), axis=1
        )  # (Nxskip)xC
        s = F.reshape(s, shape=(-1, self.skip_rnn_c_dim))  # Nx(skipxC)
        return s

    def _ar_highway(self, F, x: Tensor) -> Tensor:
        ar_x = F.slice_axis(x, axis=2, begin=-self.ar_window, end=None)  # NCT
        ar_x = F.reshape(ar_x, shape=(-3, 0))  # (NC)xT
        ar = self.ar_fc(ar_x)
        ar = F.reshape(ar, shape=(-1, self.num_series))  # NC
        return ar

    def hybrid_forward(
        self, F, past_target: Tensor, past_observed_values: Tensor
    ) -> Tensor:
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
        r, _ = self.rnn.unroll(
            inputs=c,
            length=min(self.conv_out, self.context_length),
            layout="NTC",
            merge_outputs=True,
            begin_state=self.rnn.begin_state(
                func=F.zeros,
                dtype=self.dtype,
                batch_size=c.shape[0] if isinstance(c, mx.nd.NDArray) else 0,
            ),
        )
        r = F.squeeze(
            F.slice_axis(r, axis=1, begin=-1, end=None), axis=1
        )  # NC
        s = self._skip_rnn_layer(F, c)
        fc = self.fc(F.concat(r, s, dim=1))
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
        Computes the training loss for LSTNet for multivariate time-series.
        All input tensors have NCT layout.

        Parameters
        ----------
        F
        past_target: (batch_size, num_series, context_length)
        past_observed_values: (batch_size, num_series, context_length)
        future_target: (batch_size, num_series, prediction_length)

        Returns
        -------
        Tensor
            Loss value of shape (1,)
        """

        ret = super().hybrid_forward(F, past_target, past_observed_values)
        # get the last time horizon
        future_target = F.slice_axis(future_target, axis=2, begin=-1, end=None)
        loss = F.mean(self.loss_fn(ret, future_target), axis=-1)
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
        past_target: (batch_size, num_series, context_length)
        past_observed_values: (batch_size, num_series, context_length)

        Returns
        -------
        Tensor
            Predicted samples
        """

        ret = super().hybrid_forward(F, past_target, past_observed_values)
        # (num_samples, prediction_length, target_dim)
        ret = ret.expand_dims(axis=1).expand_dims(axis=2)
        return ret
