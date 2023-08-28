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
from torch.nn.functional import linear

from .base import Attention
from .posemb import SinusoidalPositionalEmbedding


class InterAttention(Attention):
    """
    Inter-attention module with k,v from source and q from target

    Parameters
    ----------
    d_query : int
        query/decoder dimension
    d_kv: int
        key-value/encoder dimension
    n_head : int, optional
        number of attention heads, by default 1
    bias : bool, optional
        add bias term in input and output projections, by default True
    apply_kv_proj : bool, optional
        if True, apply linear projection to kv tensor and d_hidden = d_query,
            attention is performed in query feature space;
        if False, no linear porjection is applied and d_hidden = d_kv,
            attention is performed in key-value feature space;
            note that if n_head=1 in this case, it reduces to Luong's dot attention;
        by default True
    apply_rel_dist : Optional[str], optional
        add relative distance embeddings to dot-product attention, can be
            'add' (linearly combine key and dist),
            'dot' (dot product between key and dist),
            or None (disabled),
        by default None
    share_values : bool, optional
        if True, a value reprensentation is shared by all attention heads;
        only works when apply_kv_proj is False;
        by default False
        ref. https://arxiv.org/abs/1912.09363
    dropout : float, optional
        dropout rate, by default 0.0
    temperature : float, optional
        softmax temperature, by default 1.0
    """

    def __init__(
        self,
        d_query: int,
        d_kv: int,
        n_head: int = 1,
        bias: bool = True,
        apply_kv_proj: bool = True,
        apply_rel_dist: Optional[str] = None,
        share_values: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super(InterAttention, self).__init__(
            d_hidden=d_query if apply_kv_proj else d_kv,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            temperature=temperature,
        )
        self.apply_kv_proj = apply_kv_proj
        self.apply_rel_dist = apply_rel_dist
        self.share_values = share_values

        self._query_proj_weight = nn.Parameter(Tensor(self.d_hidden, d_query))
        if self.apply_kv_proj:
            self._kv_proj_weight = nn.Parameter(
                Tensor(2 * self.d_hidden, d_kv)
            )
            self._out_proj_weight = nn.Parameter(
                Tensor(self.d_hidden, self.d_hidden)
            )
        else:
            self.register_parameter("_kv_proj_weight", None)
            self.register_parameter("_out_proj_weight", None)
        if self.bias:
            self._query_proj_bias = nn.Parameter(Tensor(self.d_hidden))
            if self.apply_kv_proj:
                self._kv_proj_bias = nn.Parameter(Tensor(2 * self.d_hidden))
                self._out_proj_bias = nn.Parameter(Tensor(self.d_hidden))
            else:
                self.register_parameter("_kv_proj_bias", None)
                self.register_parameter("_out_proj_bias", None)
        else:
            self.register_parameter("_query_proj_bias", None)
            self.register_parameter("_kv_proj_bias", None)
            self.register_parameter("_out_proj_bias", None)
        if self.apply_rel_dist is None:
            self.posemb = None
            self.register_parameter("_pos_proj_weight", None)
        else:
            assert self.apply_rel_dist in ["dot", "add"]
            self.posemb = SinusoidalPositionalEmbedding(self.d_hidden)
            self._pos_proj_weight = nn.Parameter(
                Tensor(self.d_hidden, self.d_hidden)
            )
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
        init.xavier_uniform_(self._query_proj_weight)
        if self.apply_kv_proj:
            if self.share_values:
                d_kv = self._kv_proj_weight.size(1)
                del self._kv_proj_weight
                self._key_proj_weight = nn.Parameter(
                    Tensor(self.d_hidden, d_kv)
                )
                self._value_proj_weight = nn.Parameter(
                    Tensor(self.d_head, d_kv)
                )
                init.xavier_uniform_(self._key_proj_weight)
                init.xavier_uniform_(self._value_proj_weight)
                self._kv_proj_weight = pt.cat(
                    [
                        self._key_proj_weight,
                        self._value_proj_weight.repeat(self.n_head, 1),
                    ],
                    dim=0,
                )
            else:
                init.xavier_uniform_(self._kv_proj_weight)
            init.xavier_uniform_(self._out_proj_weight)
        if self.bias:
            init.zeros_(self._query_proj_bias)
            if self.apply_kv_proj:
                init.zeros_(self._kv_proj_bias)
                init.zeros_(self._out_proj_bias)
        if self.apply_rel_dist is not None:
            init.xavier_uniform_(self._pos_proj_weight)
            if self.apply_rel_dist == "add":
                init.xavier_uniform_(self._content_bias_weight)
                init.xavier_uniform_(self._position_bias_weight)

    def _compute_qkv(
        self, query: Tensor, src: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # [batch, qlen, d_hidden]
        q = linear(query, self._query_proj_weight, self._query_proj_bias)
        q = self._split_head(q)
        if self.apply_kv_proj:
            # [batch, klen, 2 * d_hidden]
            kv = linear(src, self._kv_proj_weight, self._kv_proj_bias)
            k, v = kv.chunk(2, dim=-1)
            k = self._split_head(k)
            v = self._split_head(v)
        else:
            k = v = self._split_head(src.clone())
        return q, k, v

    def _apply_mask(self, score: Tensor, key_mask: Optional[BoolTensor]):
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(dim=1)  # head
            key_mask = key_mask.unsqueeze(dim=2)  # query
            score = score.masked_fill(key_mask, float("-inf"))
        return score

    def _compute_attn_score(
        self, q: Tensor, k: Tensor, key_mask: Optional[BoolTensor], gap: int
    ) -> Tensor:
        score = q.matmul(k.transpose(-1, -2))
        if self.apply_rel_dist:
            qlen = q.size(2)
            klen = k.size(2)
            # idx[i][j] = i-j+klen
            idx = (
                pt.arange(qlen).view(-1, 1)
                - pt.arange(klen).view(1, -1)
                + klen
                + (gap - 1)
            )
            idx = idx.to(device=q.device, dtype=pt.long)
            # torch.gather requires consistent shape
            # idx.shape = [batch, n_head, qlen, klen]
            idx = idx.view(1, 1, qlen, klen).expand(
                k.size(0), k.size(1), -1, -1
            )
            # r.shape = [1, qlen+klen, d_hidden]
            r = pt.arange(qlen + klen, dtype=pt.float, device=k.device).view(
                1, -1
            )
            r = r.add_(gap)
            r = linear(self.posemb(r), self._pos_proj_weight)
            # r.shape = [1, n_head, qlen+klen-1, d_head]
            r = self._split_head(r)
            if self.apply_rel_dist == "dot":
                # s_{ij} = <r_{i-j+klen}, (q_i+k_j)>
                #        = <q_i, r_{i-j+klen}> + <r_{i-j+klen}, k_j>

                # qr_{ij} = <q_i, r_j>
                # qr'_{ij} = qr_{i,idx[i][j]} = qr_{i,i-j+klen}
                qr = q.matmul(r.transpose(-1, -2))
                qr = qr.gather(dim=-1, index=idx)
                # rk_{ij} = <r_i, k_j>
                # rk'_{ij} = rk_{idx[i][j], j} = rk_{i-j+klen, j}
                rk = r.matmul(k.transpose(-1, -2))
                rk = rk.gather(dim=-2, index=idx)
                # s_{ij} = qr_{i,i-j+klen} + rk_{i-j+klen,j}
                s = qr + rk
            else:
                # transformer-xl style: https://arxiv.org/abs/1901.02860
                # s_{ij} = <q_i, r_{i-j+klen}> + <u, k_j> + <v, r_{i-j+klen}>
                #      u = _content_bias_weight
                #      v = _position_bias_weight

                # qr_{ij} = <q_i, r_j>
                # qr'_{ij} = qr_{i,idx[i][j]} = qr_{i,i-j+klen}
                qr = q.matmul(r.transpose(-1, -2))
                qr = qr.gather(dim=-1, index=idx)
                # rk_{ij} = <v, r_i> + <u, k_j>
                # rk'_{ij} = rk_{idx[i][j], j} = rk_{i-j+klen, j}
                rk = r.matmul(
                    self._content_bias_weight.transpose(-1, -2)
                ) + self._position_bias_weight.matmul(k.transpose(-1, -2))
                rk = rk.gather(dim=-2, index=idx)
                # s_{ij} = qr_{i,i-j+klen} + rk_{i-j+klen, j}
                s = qr + rk
            # add relative positional bias to content-based attention score
            score = score + s

        score = self._apply_mask(score, key_mask)
        score = score.div(math.sqrt(self.d_head)).div(self.temperature)
        score = score.softmax(dim=-1)
        score = self.dropout(score)
        return score

    def _compute_attn_output(self, score: Tensor, v: Tensor) -> Tensor:
        # values [batch, qlen, d_hidden]
        v = self._merge_head(score.matmul(v))
        if self.apply_kv_proj:
            v = linear(v, self._out_proj_weight, self._out_proj_bias)
        return v

    def forward(
        self,
        query: Tensor,
        src: Tensor,
        src_mask: Optional[BoolTensor] = None,
        gap: int = 1,
    ) -> Tensor:
        q, k, v = self._compute_qkv(query, src)
        score = self._compute_attn_score(q, k, key_mask=src_mask, gap=gap)
        v = self._compute_attn_output(score, v)
        return v
