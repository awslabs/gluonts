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


from typing import Optional, List, Tuple, Union

import numpy as np
import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F

from tslib.nn.transformer import PositionwiseFFN
from .kernel import AttentionKernel
from .disc import SimpleDiscriminator


class EncoderModule(nn.Module):
    def __init__(
        self,
        d_data: int,
        d_feats: int,
        d_hidden: int,
        d_value: int,
        *window_size: int,
        tie_casts: bool = False,
    ) -> None:
        super(EncoderModule, self).__init__()
        self.d_data = d_data
        self.d_feats = d_feats
        self.d_hidden = d_hidden
        self.window_size = window_size
        self.tie_casts = tie_casts
        assert len(self.window_size) <= self.d_hidden

        d_shape = [d_hidden // len(window_size)] * len(window_size)
        d_shape[-1] += self.d_hidden % len(window_size)
        self._shape_weights = nn.ParameterList()
        self._shape_biases = nn.ParameterList()
        for d_s, wsize in zip(d_shape, window_size):
            self._shape_weights.append(
                nn.Parameter(Tensor(d_s, d_data + d_feats, wsize))
            )
            self._shape_biases.append(nn.Parameter(Tensor(d_s)))

        self._inter_weight = nn.Parameter(Tensor(d_value, d_data + d_feats))
        self._inter_bias = nn.Parameter(Tensor(d_value))
        if self.tie_casts:
            self._extra_weight = self._inter_weight
            self._extra_bias = self._inter_bias
        else:
            self._extra_weight = nn.Parameter(
                Tensor(d_value, d_data + d_feats)
            )
            self._extra_bias = nn.Parameter(Tensor(d_value))

        self._reset_parameters()

    def _reset_parameters(self):
        for _shape_weight in self._shape_weights:
            init.xavier_uniform_(_shape_weight)
        for _shape_bias in self._shape_biases:
            init.zeros_(_shape_bias)
        init.xavier_uniform_(self._inter_weight)
        init.zeros_(self._inter_bias)
        if not self.tie_casts:
            init.xavier_uniform_(self._extra_weight)
            init.zeros_(self._extra_bias)

    def _compute_shape(self, data: Tensor):
        s = data.transpose(1, 2)
        shape = []
        for i, wsize in enumerate(self.window_size):
            # left_pad + right_pad = window_size - 1
            left_pad = wsize // 2
            right_pad = wsize // 2 - (1 - wsize % 2)
            x = F.pad(
                s,
                [left_pad, right_pad],
                mode="constant",
                value=0.0,
            )
            x = F.conv1d(x, self._shape_weights[i], self._shape_biases[i])
            shape.append(x)
        shape = pt.cat(shape, dim=1)
        shape = shape.transpose(1, 2)
        return shape

    def _compute_value(self, data: Tensor):
        inter_value = F.linear(data, self._inter_weight, self._inter_bias)
        extra_value = F.linear(
            data[:, max(self.window_size) :],
            self._extra_weight,
            self._extra_bias,
        )
        return inter_value, extra_value

    def forward(
        self,
        data: Tensor,
        feats: Optional[Tensor],
    ):
        if feats is None:
            assert self.d_feats == 0
        else:
            assert self.d_feats == feats.size(2)
            data = pt.cat([data, feats], dim=2)
        shape = self._compute_shape(data)
        inter_value, extra_value = self._compute_value(data)
        return shape, inter_value, extra_value


class DecoderModule(nn.Module):
    def __init__(
        self,
        d_data: int,
        d_hidden: int,
    ) -> None:
        super(DecoderModule, self).__init__()
        self.d_data = d_data
        self.d_hidden = d_hidden

        self.weight = nn.Parameter(Tensor(self.d_data, self.d_hidden))
        self.bias = nn.Parameter(Tensor(d_data))

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, data: Tensor):
        return F.linear(data, self.weight, self.bias)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        kernel: AttentionKernel,
        decoder: DecoderModule,
    ) -> None:
        super(AttentionBlock, self).__init__()
        self.encoder = encoder
        self.kernel = kernel
        self.decoder = decoder
        self.window_size = max(self.encoder.window_size)

        self.register_buffer("shape", None, persistent=False)
        self.register_buffer("query", None, persistent=False)
        self.register_buffer("key", None, persistent=False)
        self.register_buffer("inter_score", None, persistent=False)
        self.register_buffer("extra_score", None, persistent=False)
        self.register_buffer("inter_value", None, persistent=False)
        self.register_buffer("extra_value", None, persistent=False)

    def forward(
        self,
        data: Tensor,
        feats: Optional[Tensor],
        mask: Optional[BoolTensor],
    ) -> Tuple[Tensor, Tensor]:
        shape, inter_value, extra_value = self.encoder(data, feats)
        self.shape = shape.detach()
        self.inter_value = inter_value.detach()
        self.extra_value = extra_value.detach()
        interp, extrap = self.kernel(shape, inter_value, extra_value, mask)
        self.query = self.kernel._query
        self.key = self.kernel._key
        self.inter_score = self.kernel._inter_score
        self.extra_score = self.kernel._extra_score
        interp = self.decoder(interp)
        extrap = self.decoder(extrap)
        return interp, extrap


class AdversarialBlock(AttentionBlock):
    def __init__(
        self,
        encoder: EncoderModule,
        kernel: AttentionKernel,
        decoder: DecoderModule,
        disc: SimpleDiscriminator,
    ) -> None:
        super(AdversarialBlock, self).__init__(encoder, kernel, decoder)
        self.disc = disc
        self.generative()
        self.register_buffer("prob_domain", None, persistent=False)

    def generative(self) -> None:
        self._generative = True
        self.disc.requires_grad_(False)
        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.kernel.requires_grad_(True)

    def discriminative(self) -> None:
        self._generative = False
        self.disc.requires_grad_(True)
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.kernel.requires_grad_(False)

    def forward(
        self,
        data: Tensor,
        feats: Optional[Tensor],
        mask: Optional[BoolTensor],
    ):
        shape, inter_value, extra_value = self.encoder(data, feats)
        self.shape = shape.detach()
        self.inter_value = inter_value.detach()
        self.extra_value = extra_value.detach()
        interp, extrap = self.kernel(shape, inter_value, extra_value, mask)
        self.query = self.kernel._query.detach()
        self.key = self.kernel._key.detach()
        self.inter_score = self.kernel._inter_score
        self.extra_score = self.kernel._extra_score

        # TODO: disc both queries and keys
        features = self.kernel._query
        if not self._generative:
            features = features.detach()
        self.prob_domain = self.disc(features)

        interp = self.decoder(interp)
        extrap = self.decoder(extrap)
        return interp, extrap
