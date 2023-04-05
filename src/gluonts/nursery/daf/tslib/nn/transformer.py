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


from typing import Optional, List
from collections import OrderedDict
import warnings

import torch as pt
from torch import Tensor, BoolTensor
from torch import nn

from .attention import (
    SelfAttention,
    GroupSelfAttention,
    CompositeAttention,
    InterAttention,
)
from .activations import Swish

add_weightnorm = False
if add_weightnorm:
    from .weightnorm import Linear, Conv1d
else:
    Linear = nn.Linear
    Conv1d = nn.Conv1d


class PositionwiseFFN(nn.Module):
    """
    Positionwise feedforward network in transformers
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        pre_ln: bool = False,
        dropout: float = 0.0,
    ):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(d_hidden, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_ln = pre_ln

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_ln:
            y = self.layer_norm(x)
        else:
            y = x
        y = self.linear1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        y = y + x
        if not self.pre_ln:
            y = self.layer_norm(y)
        return y


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_qk: int,
        d_value: int,
        d_ffn: int,
        n_head: int,
        n_groups: int = 1,
        bidirectional: bool = False,
        apply_rel_dist: Optional[str] = None,
        composite_attention: bool = False,
        pre_ln: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super(TransformerEncoderBlock, self).__init__()
        if n_groups < 1:
            raise ValueError(
                f"n_groups must be a positive integer, but {n_groups} is given"
            )
        elif n_groups == 1:
            if d_qk != d_value:
                warning.warn(
                    "When n_groups=1, `d_qk` and `d_value` is suggested to be equal, "
                    f"but are {d_qk} and {d_value}, respectively. "
                    "d_qk will be ignored."
                )
            self.shape_qk = self.composite_attention = False
            self.attention = SelfAttention(
                d_hidden=d_value,
                n_head=n_head,
                bidirectional=bidirectional,
                apply_rel_dist=apply_rel_dist,
                dropout=dropout,
                temperature=temperature,
            )
        else:
            self.shape_qk = True
            self.composite_attention = composite_attention
            Attention = (
                CompositeAttention
                if composite_attention
                else GroupSelfAttention
            )
            self.attention = Attention(
                d_qk=d_qk,
                d_value=d_value,
                n_head=n_head,
                n_groups=n_groups,
                bidirectional=bidirectional,
                apply_rel_dist=apply_rel_dist,
                dropout=dropout,
                temperature=temperature,
            )
        self.pre_ln = pre_ln
        self.layer_norm = nn.LayerNorm(d_value)
        self.dropout = nn.Dropout(dropout)
        self.ffn = PositionwiseFFN(
            d_model=d_value, d_hidden=d_ffn, dropout=dropout, pre_ln=pre_ln
        )

    def forward(
        self,
        value: Tensor,
        shape: Optional[pt.Tensor] = None,
        estimate: Optional[pt.Tensor] = None,
        mask: Optional[BoolTensor] = None,
    ) -> Tensor:
        if self.shape_qk and shape is None:
            raise ValueError("A separated shape input is required.")
        if self.composite_attention and estimate is None:
            raise ValueError("A separated estimate input is required")
        if self.pre_ln:
            x = self.layer_norm(value)
        else:
            x = value
        if not self.shape_qk:
            x = self.attention(x, mask=mask)
        else:
            if shape is None:
                raise ValueError(
                    "if shape attention is enabled, "
                    "a shape tensor must be provided."
                )
            if not self.composite_attention:
                x = self.attention(x, shape, mask=mask)
            else:
                if estimate is None:
                    raise ValueError(
                        "if composite attention is enabled, "
                        "an estimate tensor must be provided."
                    )
                x = self.attention(x, shape, estimate, mask=mask)
        x = self.dropout(x)
        value = value + x
        if not self.pre_ln:
            value = self.layer_norm(value)
        value = self.ffn(value)
        return value


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_query: int,
        d_src: int,
        d_ffn: int,
        n_head: int,
        bidirectional: bool = False,
        apply_rel_dist: Optional[str] = None,
        pre_ln: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super(TransformerDecoderBlock, self).__init__()
        self.pre_ln = pre_ln
        self.self_attention = SelfAttention(
            d_query,
            n_head,
            bidirectional=bidirectional,
            apply_rel_dist=apply_rel_dist,
            dropout=dropout,
            temperature=temperature,
        )
        if pre_ln:
            self.self_attn_layer_norm = nn.LayerNorm(d_query)
        else:
            self.self_attn_layer_norm = nn.LayerNorm(d_query)
        self.inter_attention = InterAttention(
            d_query,
            d_src,
            n_head,
            apply_kv_proj=True,
            apply_rel_dist=apply_rel_dist,
            dropout=dropout,
        )
        self.inter_attn_layer_norm = nn.LayerNorm(d_query)
        self.ffn = PositionwiseFFN(
            d_model=d_query,
            d_hidden=d_ffn,
            dropout=dropout,
            pre_ln=pre_ln,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        src: Tensor,
        mask: Optional[BoolTensor] = None,
        src_mask: Optional[BoolTensor] = None,
    ):
        if self.pre_ln:
            y = self.self_attn_layer_norm(x)
        else:
            y = x
        y = self.self_attention(y, mask=mask)
        y = self.dropout(y)
        x = y + x
        if not self.pre_ln:
            x = self.self_attn_layer_norm(x)
            y = x
        else:
            y = self.inter_attn_layer_norm(x)
        y = self.inter_attention(y, src, src_mask=src_mask)
        y = self.dropout(y)
        x = y + x
        if not self.pre_ln:
            x = self.inter_attn_layer_norm(x)
        x = self.ffn(x)
        return x
