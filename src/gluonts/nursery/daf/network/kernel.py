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
import math

import numpy as np
import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F

from tslib.nn.attention import Attention
from tslib.nn.attention.posemb import SinusoidalPositionalEmbedding


class InstanceNorm1d(nn.InstanceNorm1d):
    def forward(self, x: Tensor):
        x = x.transpose(1, 2)
        x = super(InstanceNorm1d, self).forward(x)
        x = x.transpose(1, 2)
        return x


class AttentionKernel(Attention):
    """
    Multi-Head Self-Attention Module.

    A kernel function of queries and keys, which is projection of sequence encodings

    Parameters
    ----------------
    d_hidden: int
        hidden size
    n_head: int
        number of attention heads
    n_enc_layer: int
        number of MLP hidden layers in query/key projections
    symmetric: bool, by default False
        if True, queries and keys are produced by a shared module
    share_values: bool, by default False
        if True, attention heads operate on the same value encodings
    dropout: float, by default 0.0
        dropout rate of normalized attention weights
    temperature: float, by default 1.0
        softmax temperature for attention weight normalization

    Args
    ------------------
    shape: Tensor [N, *, d_hidden]
        sequence encodings for queries and keys
    inter_value: Tensor [N, *, d_hidden] or [N, *, d_head]
        value encodings for interpolation
    extra_value: Tensor [N, *, d_hidden] or [N, *, d_head]
        value encodings for extrapolation
    mask: BoolTensor [N, *]
        mask over input sequence (1 for masked position)
    """

    def __init__(
        self,
        d_hidden: int,
        n_head: int,
        n_enc_layer: int,
        n_dec_layer: int,
        m_output: int = 2,
        symmetric: bool = False,
        share_values: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
        norm: str = "none",
        add_dist: bool = True,
    ) -> None:
        super(AttentionKernel, self).__init__(
            d_hidden=d_hidden,
            n_head=n_head,
            bias=False,
            dropout=dropout,
            temperature=temperature,
        )
        self.d_interm = self.d_hidden * m_output
        self.n_enc_layer = n_enc_layer
        self.n_dec_layer = n_dec_layer
        self.share_values = share_values
        self.symmetric = symmetric
        self.bandwidth = math.sqrt(self.d_head)
        self.add_dist = add_dist

        self._query_weights = nn.ParameterList()
        self._query_biases = nn.ParameterList() if self.bias else []
        self._key_weights = nn.ParameterList()
        self._key_biases = nn.ParameterList() if self.bias else []

        for layer in range(n_enc_layer + 1):
            _query_weight = nn.Parameter(Tensor(d_hidden, d_hidden))
            self._query_weights.append(_query_weight)
            self._key_weights.append(
                _query_weight
                if self.symmetric
                else nn.Parameter(Tensor(d_hidden, d_hidden)),
            )
            if self.bias:
                _query_bias = nn.Parameter(Tensor(d_hidden))
                self._query_biases.append(_query_bias)
                self._key_biases.append(
                    _query_bias
                    if self.symmetric
                    else nn.Parameter(Tensor(d_hidden)),
                )
            else:
                self._query_biases.append(None)
                self._key_biases.append(None)

        self._output_weights = nn.ParameterList()
        self._output_biases = nn.ParameterList() if self.bias else []
        for layer in range(n_dec_layer):
            d_input = self.d_hidden if layer == 0 else self.d_interm
            d_output = (
                self.d_hidden if layer == n_dec_layer - 1 else self.d_interm
            )
            self._output_weights.append(
                nn.Parameter(Tensor(d_output, d_input))
            )
            self._output_biases.append(
                nn.Parameter(Tensor(d_output)) if self.bias else None
            )

        if norm == "none":
            # self.enc_norm = nn.Identity()
            self.attn_norm = nn.Identity()
            self.dec_norm = nn.Identity()
        elif norm == "layer":
            # self.enc_norm = nn.LayerNorm(self.d_hidden, elementwise_affine=False)
            self.attn_norm = nn.LayerNorm(
                self.d_hidden, elementwise_affine=False
            )
            self.dec_norm = nn.LayerNorm(
                self.d_hidden, elementwise_affine=False
            )
        elif norm == "instance":
            # self.enc_norm = nn.InstanceNorm1d(self.d_hidden, affine=False)
            self.attn_norm = InstanceNorm1d(self.d_hidden, affine=False)
            self.dec_norm = InstanceNorm1d(self.d_hidden, affine=False)
        else:
            raise ValueError(f"Invalid normalization type: {norm}")

        if self.add_dist:
            self.distance_emb = SinusoidalPositionalEmbedding(d_hidden)
            self.distance_weight = nn.Parameter(Tensor(d_hidden, d_hidden))
            if self.bias:
                self.distance_bias = nn.Parameter(Tensor(d_hidden))
            else:
                self.register_parameter("distance_bias", None)
            self.distance_key = nn.Parameter(Tensor(1, n_head, self.d_head, 1))

        self._reset_parameters()

        self.register_buffer("_query", None, persistent=False)
        self.register_buffer("_key", None, persistent=False)
        self.register_buffer("_inter_score", None, persistent=False)
        self.register_buffer("_extra_score", None, persistent=False)
        self.register_buffer(
            "_inter_dist_comp", pt.zeros(self.d_head), persistent=False
        )
        self.register_buffer(
            "_extra_dist_comp", pt.zeros(self.d_head), persistent=False
        )

    def _reset_parameters(self):
        for layer in range(self.n_enc_layer + 1):
            init.xavier_uniform_(self._query_weights[layer])
            if not self.symmetric:
                init.xavier_uniform_(self._key_weights[layer])
            if self.bias:
                init.zeros_(self._query_biases[layer])
                if not self.symmetric:
                    init.zeros_(self._key_biases[layer])
        for layer in range(self.n_dec_layer):
            init.xavier_normal_(self._output_weights[layer])
            if self.bias:
                init.zeros_(self._output_biases[layer])
        if self.add_dist:
            init.xavier_normal_(self.distance_key)
            init.xavier_normal_(self.distance_weight)
            if self.bias:
                init.zeros_(self.distance_bias)

    def _kernel(
        self,
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def _compute_attn_score(
        self,
        shape: Tensor,
        left_pad: int,
        right_pad: int,
        mask: Optional[BoolTensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        q = k = shape
        for layer in range(self.n_enc_layer):
            q = F.linear(
                q, self._query_weights[layer], self._query_biases[layer]
            )
            k = F.linear(k, self._key_weights[layer], self._key_biases[layer])
            q = F.relu(q)
            k = F.relu(k)
        q = F.linear(
            q,
            self._query_weights[self.n_enc_layer],
            self._query_biases[self.n_enc_layer],
        )
        k = F.linear(
            k,
            self._key_weights[self.n_enc_layer],
            self._key_biases[self.n_enc_layer],
        )
        self._query = q
        self._key = k
        q = self._split_head(q)
        k = self._split_head(k)
        score = self._kernel(q, k)

        if self.add_dist:
            seq_len = shape.size(1)
            p = pt.arange(
                seq_len, dtype=shape.dtype, device=shape.device
            ).view(1, -1)
            p = self.distance_emb(p)
            p = F.linear(p, self.distance_weight, self.distance_bias)
            p = self._split_head(p)

            idx = pt.arange(seq_len, dtype=pt.long, device=shape.device)
            idx = pt.abs(idx.view(-1, 1) - idx.view(1, -1))
            idx = idx.view(1, 1, seq_len, seq_len).expand_as(score)

            qp = pt.matmul(q, p.transpose(-1, -2))
            qp = pt.gather(qp, dim=-1, index=idx)
            pk = pt.matmul(p, k.transpose(-1, -2))
            pk = pt.gather(pk, dim=-2, index=idx)
            vp = pt.matmul(p, self.distance_key).expand_as(score)
            vp = pt.gather(vp, dim=-2, index=idx)
            score = score + qp + pk + vp

        _inter_mask = pt.eye(
            score.size(-1), dtype=pt.bool, device=score.device
        )
        _inter_score = score.masked_fill(_inter_mask, -math.inf)
        if mask is not None:
            _key_mask = mask.unsqueeze(dim=1).unsqueeze(dim=2)
            _inter_score = _inter_score.masked_fill(_key_mask, -math.inf)
        # remove keys with padding
        _extra_score = score[
            ..., left_pad + 1 : -right_pad, left_pad : -(right_pad + 1)
        ]
        _extra_mask = pt.ones_like(_extra_score[0, 0], dtype=pt.bool).triu(
            diagonal=1
        )
        _extra_score = _extra_score.masked_fill(_extra_mask, -math.inf)
        if mask is not None:
            _key_mask = mask[..., (left_pad + right_pad + 1) :]
            _key_mask = _key_mask.unsqueeze(dim=1).unsqueeze(dim=2)
            _extra_score = _extra_score.masked_fill(_key_mask, -math.inf)
        inter_score = pt.softmax(_inter_score / self.temperature, dim=-1)
        inter_score = pt.masked_fill(inter_score, pt.isnan(inter_score), 0.0)
        self._inter_score = inter_score.detach()
        inter_score = self.dropout(inter_score)
        extra_score = pt.softmax(_extra_score / self.temperature, dim=-1)
        extra_score = pt.masked_fill(extra_score, pt.isnan(extra_score), 0.0)
        self._extra_score = extra_score.detach()
        extra_score = self.dropout(extra_score)

        if self.add_dist:
            p = p.unsqueeze(dim=-2).expand(*score.shape, self.d_head)
            idx = idx.unsqueeze(dim=-1).expand(*score.shape, self.d_head)
            p = pt.gather(p, dim=-2, index=idx)
            inter_dist_comp = pt.sum(inter_score.unsqueeze(dim=-1) * p, dim=-2)
            p = p[
                ..., left_pad + 1 : -right_pad, left_pad : -(right_pad + 1), :
            ]
            extra_dist_comp = pt.sum(extra_score.unsqueeze(dim=-1) * p, dim=-2)
            self._inter_dist_comp = inter_dist_comp.detach()
            self._extra_dist_comp = extra_dist_comp.detach()
        else:
            inter_dist_comp = self._inter_dist_comp
            extra_dist_comp = self._extra_dist_comp

        return inter_score, extra_score, inter_dist_comp, extra_dist_comp

    def _compute_attn_output(
        self,
        score: Tensor,
        value: Tensor,
        dist_comp: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.share_values:
            assert value.size(2) == self.d_head
            value = pt.unsqueeze(value, dim=1)
        else:
            assert value.size(2) == self.d_hidden
            value = self._split_head(value)
        output = score @ value
        output = output + dist_comp
        output = self._merge_head(output)
        output = self.attn_norm(output)
        hidden = output
        for layer in range(self.n_dec_layer):
            hidden = F.linear(
                hidden, self._output_weights[layer], self._output_biases[layer]
            )
            if layer < self.n_dec_layer - 1:
                hidden = F.gelu(hidden)
                hidden = self.dropout(hidden)
            elif layer == self.n_dec_layer - 1:
                output = hidden + output
                output = self.dec_norm(output)
        return output

    def forward(
        self,
        shape: Tensor,
        inter_value: Tensor,
        extra_value: Tensor,
        mask: Optional[BoolTensor],
    ) -> Tuple[Tensor, Tensor]:
        window_size = inter_value.size(1) - extra_value.size(1)
        left_pad = window_size // 2
        right_pad = window_size // 2 - (1 - window_size % 2)
        (
            inter_score,
            extra_score,
            inter_dist_comp,
            extra_dist_comp,
        ) = self._compute_attn_score(shape, left_pad, right_pad, mask)
        interp = self._compute_attn_output(
            inter_score, inter_value, inter_dist_comp
        )
        extrap = self._compute_attn_output(
            extra_score, extra_value, extra_dist_comp
        )
        return interp, extrap


class ExpKernel(AttentionKernel):
    def _kernel(
        self,
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        return pt.matmul(q, k.transpose(-1, -2)) / self.bandwidth


class RBFKernel(AttentionKernel):
    def _kernel(
        self,
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        q = q.unsqueeze(dim=-2)
        k = k.unsqueeze(dim=-3)
        score = -pt.sum(pt.pow(q - k, 2), dim=-1) / self.bandwidth
        return score
