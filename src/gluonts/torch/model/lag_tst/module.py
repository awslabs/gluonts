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

from typing import Tuple, Optional, List

import numpy as np
from pandas.tseries.frequencies import to_offset
import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.util import unsqueeze_expand, lagged_sequence_values
from gluonts.time_feature import norm_freq_str
from gluonts.time_feature.lag import _make_lags


def get_lags_for_frequency(
    freq_str: str, lag_ub: int = 1200, num_lags: Optional[int] = None
) -> List[int]:
    """
    Generates a list of lags that that are appropriate for the given frequency
    string.

    By default all frequencies have the following lags: [1].
    Remaining lags correspond to the same `season` (+/- `delta`) in previous
    `k` cycles. Here `delta` and `k` are chosen according to the existing code.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H",
        "5min", "1D" etc.

    lag_ub
        The maximum value for a lag.

    num_lags
        Maximum number of lags; by default all generated lags are returned
    """

    # Lags are target values at the same `season` (+/- delta) but in the
    # previous cycle.
    def _make_lags_for_second(multiple, num_cycles=3):
        # We use previous ``num_cycles`` hours to generate lags
        return [
            _make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_minute(multiple, num_cycles=3):
        # We use previous ``num_cycles`` hours to generate lags
        return [
            _make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_hour(multiple, num_cycles=7):
        # We use previous ``num_cycles`` days to generate lags
        return [
            _make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_day(
        multiple, num_cycles=4, days_in_week=7, days_in_month=30
    ):
        # We use previous ``num_cycles`` weeks to generate lags
        # We use the last month (in addition to 4 weeks) to generate lag.
        return [
            _make_lags(k * days_in_week // multiple, 1)
            for k in range(1, num_cycles + 1)
        ] + [_make_lags(days_in_month // multiple, 1)]

    def _make_lags_for_week(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        # Additionally, we use previous 4, 8, 12 weeks
        return [
            _make_lags(k * 52 // multiple, 1) for k in range(1, num_cycles + 1)
        ] + [[4 // multiple, 8 // multiple, 12 // multiple]]

    def _make_lags_for_month(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        return [
            _make_lags(k * 12 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    # multiple, granularity = get_granularity(freq_str)
    offset = to_offset(freq_str)
    # normalize offset name, so that both `W` and `W-SUN` refer to `W`
    offset_name = norm_freq_str(offset.name)

    if offset_name == "A":
        lags = []
    elif offset_name == "Q":
        assert (
            offset.n == 1
        ), "Only multiple 1 is supported for quarterly. Use x month instead."
        lags = _make_lags_for_month(offset.n * 3.0)
    elif offset_name == "M":
        lags = _make_lags_for_month(offset.n)
    elif offset_name == "W":
        lags = _make_lags_for_week(offset.n)
    elif offset_name == "D":
        lags = _make_lags_for_day(offset.n) + _make_lags_for_week(
            offset.n / 7.0
        )
    elif offset_name == "B":
        lags = _make_lags_for_day(
            offset.n, days_in_week=5, days_in_month=22
        ) + _make_lags_for_week(offset.n / 5.0)
    elif offset_name == "H":
        lags = (
            _make_lags_for_hour(offset.n)
            + _make_lags_for_day(offset.n / 24)
            + _make_lags_for_week(offset.n / (24 * 7))
        )
    # minutes
    elif offset_name == "T":
        lags = (
            _make_lags_for_minute(offset.n)
            + _make_lags_for_hour(offset.n / 60)
            + _make_lags_for_day(offset.n / (60 * 24))
            + _make_lags_for_week(offset.n / (60 * 24 * 7))
        )
    # second
    elif offset_name == "S":
        lags = (
            _make_lags_for_second(offset.n)
            + _make_lags_for_minute(offset.n / 60)
            + _make_lags_for_hour(offset.n / (60 * 60))
        )
    else:
        raise Exception("invalid frequency")

    # flatten lags list and filter
    lags = [
        int(lag) for sub_list in lags for lag in sub_list if 7 < lag <= lag_ub
    ]
    lags = [1] + sorted(list(set(lags)))

    return lags[:num_lags]


def make_linear_layer(dim_in, dim_out):
    lin = nn.Linear(dim_in, dim_out)
    torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
    torch.nn.init.zeros_(lin.bias)
    return lin


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Features are not interleaved. The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        # set early to avoid an error in pytorch-1.8+
        out.requires_grad = False

        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen x ...]."""
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class LagTSTModel(nn.Module):
    """
    Module implementing the LagTST model for forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        freq: str,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_encoder_layers: int,
        scaling: str,
        lags_seq: Optional[List[int]] = None,
        distr_output=StudentTOutput(),
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.d_model = d_model
        self.distr_output = distr_output

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        # project from number of lags + 2 features (loc and scale) to d_model
        self.patch_proj = make_linear_layer(len(self.lags_seq) + 2, d_model)

        self.positional_encoding = SinusoidalPositionalEmbedding(
            self.context_length, d_model
        )

        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.flatten = nn.Linear(
            d_model * self.context_length, prediction_length * d_model
        )

        self.args_proj = self.distr_output.get_args_proj(d_model)

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self._past_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length), dtype=torch.float
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )

        lags = lagged_sequence_values(
            self.lags_seq,
            past_target_scaled[:, : -self.context_length, ...],
            past_target_scaled[:, -self.context_length :, ...],
            dim=-1,
        )

        # add loc and scale to past_target_patches as additional features
        log_abs_loc = loc.abs().log1p()
        log_scale = scale.log()
        expanded_static_feat = unsqueeze_expand(
            torch.cat([log_abs_loc, log_scale], dim=-1),
            dim=1,
            size=lags.shape[1],
        )
        inputs = torch.cat((lags, expanded_static_feat), dim=-1)

        # project patches
        enc_in = self.patch_proj(inputs)
        embed_pos = self.positional_encoding(enc_in.size())

        # transformer encoder with positional encoding
        enc_out = self.encoder(enc_in + embed_pos)

        # flatten and project to prediction length * d_model
        flatten_out = self.flatten(enc_out.flatten(start_dim=1))

        # project to distribution arguments
        distr_args = self.args_proj(
            flatten_out.reshape(-1, self.prediction_length, self.d_model)
        )
        return distr_args, loc, scale
