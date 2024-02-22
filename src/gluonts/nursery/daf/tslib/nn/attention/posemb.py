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


from typing import Optional, List, Tuple
from functools import reduce
from operator import mul, add
import math

import torch as pt
from torch import Tensor, LongTensor
from torch import nn
from torch.nn import init


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_embed: int):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_embed % 2 > 0:
            raise ValueError(
                "sinusoidal embedding must have an even dimension"
            )
        self.d_embed = d_embed
        inv_freq = pt.arange(0, self.d_embed, 2, dtype=pt.float)
        inv_freq /= self.d_embed
        inv_freq *= -math.log(1e4)
        self.register_buffer("inv_freq", inv_freq.exp_())

    def forward(self, pos_seq: LongTensor) -> Tensor:
        pos_seq = pos_seq.unsqueeze(dim=-1).float() * self.inv_freq
        return pt.cat([pos_seq.sin(), pos_seq.cos()], dim=-1)


class LearnablePositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_embed: int,
        max_len: int,
        hierarchy: Optional[List[Tuple[int, int]]] = None,
    ):
        super(LearnablePositionalEmbedding, self).__init__()
        if hierarchy is None:
            hierarchy = [(max_len, d_embed)]
        sub_shape, d_sub_embeds = zip(*hierarchy)
        assert (
            reduce(mul, sub_shape, 1) == max_len
        ), f"decomposed shape {sub_shape} inconsistent with max length {max_len}"
        assert (
            reduce(add, d_sub_embeds, 0) == d_embed
        ), f"decomposed embed dim {d_sub_embeds} inconsistent with overall embed dim {d_embed}"
        self.d_embed = d_embed
        self.d_sub_embeds = d_sub_embeds
        self.max_len = max_len
        self.sub_shape = sub_shape

        self._weights = nn.ParameterList(
            [
                nn.Parameter(Tensor(size, dim))
                for size, dim in zip(self.sub_shape, self.d_sub_embeds)
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self._weights:
            init.xavier_normal_(weight)

    @property
    def n_levels(self):
        return len(self._weights)

    def forward(self, pos_seq: LongTensor):
        x = pos_seq.clone()
        embeddings = []
        for l in range(self.n_levels):
            size = self.sub_shape[l]
            weight = self._weights[l]
            index = x.remainder(size)
            embeddings.append(nn.functional.embedding(index, weight))
            x = x.div(size)
        if pt.any(x > 0):
            exceed_idx = [t[0].item() for t in pt.where(x > 0)]
            raise ValueError(
                f"Given positional index at {exceed_idx} exceeds the max sequence length {self.max_len}"
            )
        embeddings = pt.cat(embeddings, dim=-1)
        return embeddings
