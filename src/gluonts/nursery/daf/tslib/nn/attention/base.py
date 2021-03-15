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
from torch import Tensor
from torch import nn


class Attention(nn.Module):
    """
    Base class of attention modules
    *NOTE*: d_hidden must be divisible by n_head

    Parameters
    ----------
    d_hidden : int
        hidden dimension
    n_head : int, optional
        number of attention heads, by default 1
    bias : bool, optional
        add bias term in input and output projections, by default True
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
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super(Attention, self).__init__()
        if d_hidden % n_head != 0:
            raise ValueError(
                f"hidden dim {d_hidden} cannot be split into {n_head} heads"
            )
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.d_head = d_hidden // n_head
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature

    def _split_head(self, x: Tensor) -> Tensor:
        """
        Split hidden state into multi-heads

        Args
        ----------
            x : Tensor [batch, length, d_hidden]

        Returns
        -------
            Tensor [batch, n_head, length, d_head]
        """
        return (
            x.view(*x.shape[:-1], self.n_head, -1).transpose(1, 2).contiguous()
        )

    def _merge_head(self, x: Tensor) -> Tensor:
        """
        Merge multi-heads into one hidden state

        Args
        ----------
            x : Tensor [batch, n_head, length, d_head]

        Returns
        -------
            Tensor [batch, length, d_hidden]
        """
        x = x.transpose(1, 2).contiguous()
        return x.view(*x.shape[:-2], -1)

    def _compute_qkv(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_attn_score(self, q: Tensor, k: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def _compute_attn_output(
        self, score: Tensor, v: Tensor, **kwargs
    ) -> Tensor:
        raise NotImplementedError
