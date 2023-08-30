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

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pts.modules import MeanScaler, NOPScaler


class LSTNetBase(nn.Module):
    def __init__(
        self,
        num_series: int,
        channels: int,
        kernel_size: int,
        rnn_cell_type: str,
        rnn_num_cells: int,
        skip_rnn_cell_type: str,
        skip_rnn_num_cells: int,
        skip_size: int,
        ar_window: int,
        context_length: int,
        horizon: Optional[int],
        prediction_length: Optional[int],
        dropout_rate: float,
        output_activation: Optional[str],
        scaling: bool,
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
            "GRU",
            "LSTM",
        ], "`rnn_cell_type` must be either 'GRU' or 'LSTM' "
        assert skip_rnn_cell_type in [
            "GRU",
            "LSTM",
        ], "`skip_rnn_cell_type` must be either 'GRU' or 'LSTM' "

        conv_out = context_length - kernel_size
        self.conv_skip = conv_out // skip_size
        assert self.conv_skip > 0, (
            "conv1d output size must be greater than or equal to `skip_size`\n"
            "Choose a smaller `kernel_size` or bigger `context_length`"
        )

        self.cnn = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=(num_series, kernel_size),
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_cell_type]
        self.rnn = rnn(
            input_size=channels,
            hidden_size=rnn_num_cells,
            # dropout=dropout_rate,
        )

        skip_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[skip_rnn_cell_type]
        self.skip_rnn_num_cells = skip_rnn_num_cells
        self.skip_rnn = skip_rnn(
            input_size=channels,
            hidden_size=skip_rnn_num_cells,
            # dropout=dropout_rate,
        )

        self.fc = nn.Linear(
            rnn_num_cells + skip_size * skip_rnn_num_cells, num_series
        )

        if self.horizon:
            self.ar_fc = nn.Linear(ar_window, 1)
        else:
            self.ar_fc = nn.Linear(ar_window, prediction_length)

        if scaling:
            self.scaler = MeanScaler(keepdim=True, time_first=False)
        else:
            self.scaler = NOPScaler(keepdim=True, time_first=False)

    def forward(
        self, past_target: torch.Tensor, past_observed_values: torch.Tensor
    ) -> torch.Tensor:
        scaled_past_target, scale = self.scaler(
            past_target[..., -self.context_length :],  # [B, C, T]
            past_observed_values[..., -self.context_length :],  # [B, C, T]
        )

        # CNN
        c = F.relu(self.cnn(scaled_past_target.unsqueeze(1)))
        c = self.dropout(c)
        c = c.squeeze(2)  # [B, C, T]

        # RNN
        r = c.permute(2, 0, 1)  # [F (T), B, C]
        _, r = self.rnn(r)  # [1, B, H]
        r = self.dropout(r.squeeze(0))  # [B, H]

        # Skip-RNN
        skip_c = c[..., -self.conv_skip * self.skip_size :]
        skip_c = skip_c.reshape(
            -1, self.channels, self.conv_skip, self.skip_size
        )
        skip_c = skip_c.permute(2, 0, 3, 1)
        skip_c = skip_c.reshape((self.conv_skip, -1, self.channels))
        _, skip_c = self.skip_rnn(skip_c)
        skip_c = skip_c.reshape((-1, self.skip_size * self.skip_rnn_num_cells))
        skip_c = self.dropout(skip_c)

        res = self.fc(torch.cat((r, skip_c), 1)).unsqueeze(-1)

        # Highway
        ar_x = scaled_past_target[..., -self.ar_window :]
        ar_x = ar_x.reshape(-1, self.ar_window)

        ar_x = self.ar_fc(ar_x)
        if self.horizon:
            ar_x = ar_x.reshape(-1, self.num_series, 1)
        else:
            ar_x = ar_x.reshape(-1, self.num_series, self.prediction_length)
        out = res + ar_x

        if self.output_activation is None:
            return out, scale

        return (
            (
                torch.sigmoid(out)
                if self.output_activation == "sigmoid"
                else torch.tanh(out)
            ),
            scale,
        )


class LSTNetTrain(LSTNetBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.L1Loss()

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
    ) -> torch.Tensor:
        ret, scale = super().forward(past_target, past_observed_values)

        if self.horizon:
            future_target = future_target[..., -1:]

        loss = self.loss_fn(ret * scale, future_target)
        return loss


class LSTNetPredict(LSTNetBase):
    def forward(
        self, past_target: torch.Tensor, past_observed_values: torch.Tensor
    ) -> torch.Tensor:
        ret, scale = super().forward(past_target, past_observed_values)
        ret = (ret * scale).permute(0, 2, 1)

        return ret.unsqueeze(1)
