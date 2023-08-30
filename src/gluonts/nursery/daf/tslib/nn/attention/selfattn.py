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


from typing import Tuple, Optional
import math

import torch as pt
from torch import Tensor, BoolTensor
from torch import nn
from torch.nn import init
from torch.nn.functional import linear, conv1d

from .base import Attention
from .posemb import SinusoidalPositionalEmbedding


class SelfAttention(Attention):
    """
    Self-attention module with q,k,v from the same input

    Parameters
    ----------
    d_hidden : int
        hidden dimension
    n_head : int, optional
        number of attention heads, by default 1
    bias : bool, optional
        add bias term in input and output projections, by default True
    bidirectional : bool, optional
        if False, add a mask to avoid backward attention, by default False
    apply_rel_dist : Optional[str], optional
        add relative distance embeddings to dot-product attention, can be
            'add' (linearly combine key and dist),
            'dot' (dot product between key and dist),
            or None (disabled),
        by default None
    share_values : bool, optional
        if True, a value reprensentation is shared by all attention heads, by default False
        ref. https://arxiv.org/abs/1912.09363
    dropout : float, optional
        dropout rate, by default 0.0
    temperature : float, optional
        softmax temperature, by default 1.0
    """

    def __init__(
        self,
        d_hidden: int,
        n_head: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        apply_rel_dist: Optional[str] = None,
        share_values: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super(SelfAttention, self).__init__(
            d_hidden=d_hidden,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            temperature=temperature,
        )
        self.bidirectional = bidirectional
        self.apply_rel_dist = apply_rel_dist
        self.share_values = share_values

        d_value = self.d_head if self.share_values else d_hidden
        self._qk_proj_weight = nn.Parameter(Tensor(2 * d_hidden, d_hidden))
        self._value_proj_weight = nn.Parameter(Tensor(d_value, d_hidden))
        self._out_proj_weight = nn.Parameter(Tensor(d_hidden, d_hidden))
        if bias:
            self._qk_proj_bias = nn.Parameter(Tensor(2 * d_hidden))
            self._value_proj_bias = nn.Parameter(Tensor(d_value))
            self._out_proj_bias = nn.Parameter(Tensor(d_hidden))
        else:
            self.register_parameter("_in_proj_bias", None)
            self.register_parameter("_out_proj_bias", None)

        if self.apply_rel_dist is None:
            self.posemb = None
            self.register_parameter("_pos_proj_weight", None)
        else:
            assert self.apply_rel_dist in ["dot", "add"]
            self.posemb = SinusoidalPositionalEmbedding(d_hidden)
            self._pos_proj_weight = nn.Parameter(Tensor(d_hidden, d_hidden))
            if self.apply_rel_dist == "add":
                self._content_bias_weight = nn.Parameter(
                    Tensor(n_head, 1, self.d_head)
                )
                self._position_bias_weight = nn.Parameter(
                    Tensor(n_head, 1, self.d_head)
                )
            else:
                self.register_parameter("_content_bias_weight", None)
                self.register_parameter("_position_bias_weight", None)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self._qk_proj_weight)
        init.xavier_uniform_(self._value_proj_weight)
        init.xavier_uniform_(self._out_proj_weight)
        if self.bias:
            init.zeros_(self._qk_proj_bias)
            init.zeros_(self._value_proj_bias)
            init.zeros_(self._out_proj_bias)
        if self.apply_rel_dist is not None:
            init.xavier_uniform_(self._pos_proj_weight)
            if self.apply_rel_dist == "add":
                init.xavier_uniform_(self._content_bias_weight)
                init.xavier_uniform_(self._position_bias_weight)

    def _compute_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        qk = linear(x, self._qk_proj_weight, self._qk_proj_bias)
        q, k = qk.chunk(2, dim=-1)
        q = self._split_head(q)
        k = self._split_head(k)
        v = linear(x, self._value_proj_weight, self._value_proj_bias)
        if self.share_values:
            v = pt.stack([v] * self.n_head, dim=1)
        else:
            v = self._split_head(v)
        return q, k, v

    def _apply_mask(
        self, score: Tensor, key_mask: Optional[BoolTensor]
    ) -> Tensor:
        if not self.bidirectional:
            unid_mask = score.new_ones(*score.shape[-2:], dtype=pt.bool).triu(
                diagonal=1
            )
            score = score.masked_fill(unid_mask, float("-inf"))
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(dim=1)  # head
            key_mask = key_mask.unsqueeze(dim=2)  # query
            score = score.masked_fill(key_mask, float("-inf"))
        return score

    def _compute_attn_score(
        self, q: Tensor, k: Tensor, mask: Optional[BoolTensor] = None
    ) -> Tensor:
        score = q.matmul(k.transpose(-1, -2))
        if self.apply_rel_dist is not None:
            # score_{ij} = <q_i, k_j> + s_{ij}
            klen = k.size(2)
            # idx.shape = [klen, klen]
            # idx[i][j] = i-j
            idx = pt.arange(klen, dtype=pt.long, device=k.device)
            idx = idx.view(-1, 1) - idx.view(1, -1)
            # idx[i][j] = |i-j|
            idx = idx.abs_()
            # torch.gather requires consistent shape
            # idx.shape = [batch, n_head, klen, klen]
            idx = idx.view(1, 1, klen, klen).expand(
                k.size(0), k.size(1), -1, -1
            )
            # dist representation r for attention
            # r.shape = [1, klen, d_hidden]
            r = pt.arange(klen, dtype=pt.float, device=k.device).view(1, -1)
            r = linear(self.posemb(r), self._pos_proj_weight)
            # r.shape = [1, n_head, klen, d_head]
            r = self._split_head(r)
            if self.apply_rel_dist == "dot":
                # s_{ij} = <r_{|i-j|}, (q_i+k_j)>
                #        = <q_i, r_{|i-j|}> + <r_{|i-j|}, k_j>

                # qr_{ij} = <q_i, r_j>
                # qr'_{ij} = qr_{i,idx[i][j]} = qr_{i,|i-j|}
                qr = q.matmul(r.transpose(-1, -2))
                qr = qr.gather(dim=-1, index=idx)
                # rk_{ij} = <r_i, k_j>
                # rk'_{ij} = rk_{idx[i][j], j} = rk_{|i-j|, j}
                rk = r.matmul(k.transpose(-1, -2))
                rk = rk.gather(dim=-2, index=idx)
                # s_{ij} = qr_{i,|i-j|} + rk_{|i-j|,j}
                s = qr + rk
            else:
                # transformer-xl style: https://arxiv.org/abs/1901.02860
                # s_{ij} = <q_i, r_{|i-j|}> + <u, k_j> + <v, r_{|i-j|}>
                #      u = _content_bias_weight
                #      v = _position_bias_weight

                # qr_{ij} = <q_i, r_j>
                # qr'_{ij} = qr_{i,idx[i][j]} = qr_{i,|i-j|}
                qr = q.matmul(r.transpose(-1, -2))
                qr = qr.gather(dim=-1, index=idx)
                # rk_{ij} = <v, r_i> + <u, k_j>
                # rk'_{ij} = rk_{idx[i][j], j} = rk_{|i-j|, j}
                rk = r.matmul(
                    self._content_bias_weight.transpose(-1, -2)
                ) + self._position_bias_weight.matmul(k.transpose(-1, -2))
                rk = rk.gather(dim=-2, index=idx)
                # s_{ij} = qr_{i,|i-j|} + rk_{|i-j|, j}
                s = qr + rk
            # add relative positional bias to content-based attention score
            score = score + s

        score = self._apply_mask(score, mask)
        score = score.div(math.sqrt(self.d_head)).div(self.temperature)
        score = score.softmax(dim=-1)
        score = self.dropout(score)
        return score

    def _compute_attn_output(self, score: Tensor, v: Tensor) -> Tensor:
        v = self._merge_head(score.matmul(v))
        v = linear(v, self._out_proj_weight, self._out_proj_bias)
        return v

    def forward(
        self, x: Tensor, *, mask: Optional[BoolTensor] = None
    ) -> Tensor:
        q, k, v = self._compute_qkv(x)
        score = self._compute_attn_score(q, k, mask)
        v = self._compute_attn_output(score, v)
        return v


class GroupSelfAttention(SelfAttention):
    """
    Self-attention module with q,k from the same input tensor.
    The input tensor is the concatenation of `n_groups` of slightly different feature maps.
    Thus the projections are 1x1 group convolutions.
    *NOTE*: d_qk, d_hidden, n_head must be divisible by n_groups
    """

    def __init__(
        self,
        d_qk: int,
        d_value: int,
        n_head: int = 1,
        n_groups: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        apply_rel_dist: Optional[str] = None,
        share_values: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        try:
            assert d_qk % n_groups == 0
            assert d_value % n_groups == 0
            assert n_head % n_groups == 0
        except AssertionError:
            raise ValueError(
                "d_qk, d_hidden, n_head must be divisible by n_groups"
            )
        if d_qk // n_groups > d_value:
            raise ValueError(
                "dimension of each group should not exceed d_value"
            )
        self.d_qk = d_qk
        self.d_value = d_value
        self.n_groups = n_groups

        super(GroupSelfAttention, self).__init__(
            d_hidden=d_value,
            n_head=n_head,
            bias=bias,
            bidirectional=bidirectional,
            apply_rel_dist=apply_rel_dist,
            share_values=share_values,
            dropout=dropout,
            temperature=temperature,
        )

    def _reset_parameters(self):
        del self._qk_proj_weight
        self._qk_proj_weight = nn.Parameter(
            Tensor(2 * self.d_hidden, self.d_qk // self.n_groups, 1)
        )
        super(GroupSelfAttention, self)._reset_parameters()

    def _compute_qkv(
        self, value: Tensor, shape: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        qk = (
            conv1d(
                input=shape.transpose(1, 2),
                weight=self._qk_proj_weight,
                bias=self._qk_proj_bias,
                groups=self.n_groups,
            )
            .transpose(1, 2)
            .contiguous()
        )
        qk = qk.chunk(self.n_groups * 2, dim=-1)
        q = self._split_head(pt.cat(qk[0::2], dim=-1))
        k = self._split_head(pt.cat(qk[1::2], dim=-1))
        v = linear(value, self._value_proj_weight, self._value_proj_bias)
        if self.share_values:
            v = pt.stack([v] * self.n_head, dim=1)
        else:
            v = self._split_head(v)
        return q, k, v

    def forward(
        self,
        value: Tensor,
        shape: Tensor,
        *,
        mask: Optional[BoolTensor] = None
    ) -> Tensor:
        q, k, v = self._compute_qkv(value, shape)
        score = self._compute_attn_score(q, k, mask)
        v = self._compute_attn_output(score, v)
        return v


class CompositeAttention(GroupSelfAttention):
    """
    Self-attention module with groups that has a regressive component.
    """

    def _compute_qkv(
        self, value: Tensor, shape: Tensor, estimate: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        q, k, v = super(CompositeAttention, self)._compute_qkv(value, shape)
        v = v.roll(-1, dim=2)
        e = self._split_head(e)
        return q, k, v, e

    def _compute_attn_output(
        self, score: Tensor, v: Tensor, e: Tensor
    ) -> Tensor:
        history_score = score.tril(diagonal=-1)
        current_score = score.diagonal(dim1=-2, dim2=-1).unsqueeze(dim=-1)
        v = history_score @ v + current_score * e
        v = self._merge_head(v)
        v = linear(v, self._out_proj_weight, self._out_proj_bias)
        return v

    def forward(
        self,
        value: Tensor,
        shape: Tensor,
        estimate: Tensor,
        mask: Optional[BoolTensor] = None,
    ) -> Tensor:
        q, k, v, e = self._compute_qkv(value, shape, estimate)
        score = self._compute_attn_score(q, k, mask)
        v = self._compute_attn_output(score, v, e)
        return v
