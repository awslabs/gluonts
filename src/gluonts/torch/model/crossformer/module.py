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

from math import ceil
from typing import Optional, Tuple

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import Output, StudentTOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluonts.torch.util import weighted_average

from .layers import DSWEmbedding, Encoder, Decoder


class CrossformerModel(nn.Module):
    """
    Crossformer backbone adapted to GluonTS conventions.

    The model keeps the original segment-wise encoder/decoder structure, but
    performs missing-value-aware scaling inside the module and projects the
    final horizon states through a generic GluonTS ``Output`` head.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        seg_len: int,
        win_size: int = 2,
        factor: int = 10,
        d_model: int = 64,
        d_ff: int = 128,
        n_heads: int = 4,
        num_encoder_layers: int = 3,
        num_feat_dynamic_real: int = 0,
        scaling: Optional[str] = "mean",
        distr_output: Output = StudentTOutput(),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert seg_len > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.seg_len = seg_len
        self.win_size = win_size
        self.factor = factor
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.distr_output = distr_output

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.pad_context_length = ceil(context_length / seg_len) * seg_len
        self.pad_prediction_length = ceil(prediction_length / seg_len) * seg_len
        self.context_pad = self.pad_context_length - context_length
        self.prediction_pad = self.pad_prediction_length - prediction_length
        self.in_seg_num = self.pad_context_length // seg_len
        self.out_seg_num = self.pad_prediction_length // seg_len

        enc_input_dim = seg_len + 2 + num_feat_dynamic_real * seg_len
        dec_input_dim = 2 + num_feat_dynamic_real * seg_len

        self.encoder_value_embedding = DSWEmbedding(
            input_dim=enc_input_dim, d_model=d_model
        )
        self.decoder_value_embedding = DSWEmbedding(
            input_dim=dec_input_dim, d_model=d_model
        )
        self.encoder_pos_embedding = nn.Parameter(
            torch.randn(1, 1, self.in_seg_num, d_model)
        )
        self.decoder_pos_embedding = nn.Parameter(
            torch.randn(1, 1, self.out_seg_num, d_model)
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.encoder = Encoder(
            num_blocks=num_encoder_layers,
            win_size=win_size,
            in_seg_num=self.in_seg_num,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            block_depth=1,
            dropout=dropout,
            factor=factor,
        )
        self.decoder = Decoder(
            seg_len=seg_len,
            out_seg_num=self.out_seg_num,
            num_layers=num_encoder_layers + 1,
            d_model=d_model,
            latent_dim=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            factor=factor,
        )
        self.args_proj = self.distr_output.get_args_proj(d_model)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        inputs = {
            "past_target": Input(
                shape=(batch_size, self.context_length, -1),
                dtype=torch.float,
            ),
            "past_observed_values": Input(
                shape=(batch_size, self.context_length, -1),
                dtype=torch.float,
            ),
        }
        if self.num_feat_dynamic_real > 0:
            inputs.update(
                {
                    "past_time_feat": Input(
                        shape=(
                            batch_size,
                            self.context_length,
                            self.num_feat_dynamic_real,
                        ),
                        dtype=torch.float,
                    ),
                    "future_time_feat": Input(
                        shape=(
                            batch_size,
                            self.prediction_length,
                            self.num_feat_dynamic_real,
                        ),
                        dtype=torch.float,
                    ),
                }
            )
        return InputSpec(inputs, torch.zeros)

    def _pad_sequence(
        self, x: torch.Tensor, total_length: int, left: bool
    ) -> torch.Tensor:
        if x.shape[1] >= total_length:
            return x

        pad = total_length - x.shape[1]
        filler = x[:, :1, :] if left else x[:, -1:, :]
        filler = filler.expand(-1, pad, -1)
        return torch.cat((filler, x), dim=1) if left else torch.cat((x, filler), dim=1)

    def _segmentify_target(
        self, x: torch.Tensor, total_length: int, left: bool
    ) -> torch.Tensor:
        x = self._pad_sequence(x, total_length=total_length, left=left)
        batch_size, _, target_dim = x.shape
        x = x.reshape(batch_size, total_length // self.seg_len, self.seg_len, target_dim)
        return x.permute(0, 3, 1, 2).contiguous()

    def _segmentify_feat(
        self, x: torch.Tensor, total_length: int, left: bool
    ) -> torch.Tensor:
        x = self._pad_sequence(x, total_length=total_length, left=left)
        batch_size, _, feat_dim = x.shape
        return x.reshape(batch_size, total_length // self.seg_len, self.seg_len * feat_dim)

    def _expand_feat_segments(
        self, feat_segments: torch.Tensor, target_dim: int
    ) -> torch.Tensor:
        return feat_segments.unsqueeze(1).expand(-1, target_dim, -1, -1)

    def _build_encoder_input(
        self,
        past_target_scaled: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        past_time_feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target_segments = self._segmentify_target(
            past_target_scaled, total_length=self.pad_context_length, left=True
        )
        batch_size, target_dim, seg_num, _ = target_segments.shape

        log_abs_loc = (
            loc.abs().log1p().transpose(1, 2).unsqueeze(2).expand(-1, -1, seg_num, -1)
        )
        log_scale = (
            scale.log().transpose(1, 2).unsqueeze(2).expand(-1, -1, seg_num, -1)
        )

        inputs = [target_segments, log_abs_loc, log_scale]
        if past_time_feat is not None:
            feat_segments = self._segmentify_feat(
                past_time_feat, total_length=self.pad_context_length, left=True
            )
            inputs.append(self._expand_feat_segments(feat_segments, target_dim))

        x = torch.cat(inputs, dim=-1)
        x = self.encoder_value_embedding(x)
        return self.encoder_norm(
            x + self.encoder_pos_embedding[:, :, :seg_num, :]
        )

    def _build_decoder_input(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        target_dim: int,
    ) -> torch.Tensor:
        log_abs_loc = (
            loc.abs()
            .log1p()
            .transpose(1, 2)
            .unsqueeze(2)
            .expand(-1, -1, self.out_seg_num, -1)
        )
        log_scale = (
            scale.log()
            .transpose(1, 2)
            .unsqueeze(2)
            .expand(-1, -1, self.out_seg_num, -1)
        )

        inputs = [log_abs_loc, log_scale]
        if future_time_feat is not None:
            feat_segments = self._segmentify_feat(
                future_time_feat,
                total_length=self.pad_prediction_length,
                left=False,
            )
            inputs.append(self._expand_feat_segments(feat_segments, target_dim))

        x = torch.cat(inputs, dim=-1)
        x = self.decoder_value_embedding(x)
        return self.decoder_norm(
            x + self.decoder_pos_embedding[:, :, : self.out_seg_num, :]
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )

        encoder_input = self._build_encoder_input(
            past_target_scaled=past_target_scaled,
            loc=loc,
            scale=scale,
            past_time_feat=past_time_feat,
        )
        encoder_states = self.encoder(encoder_input)

        decoder_input = self._build_decoder_input(
            loc=loc,
            scale=scale,
            future_time_feat=future_time_feat,
            target_dim=past_target.shape[-1],
        )
        latent = self.decoder(decoder_input, encoder_states)
        latent = latent[:, : self.prediction_length, :, :]

        distr_args = self.args_proj(latent)
        return distr_args, loc, scale

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distr_args, loc, scale = self(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )
        loss = self.distr_output.loss(
            target=future_target,
            distr_args=distr_args,
            loc=loc,
            scale=scale,
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)
