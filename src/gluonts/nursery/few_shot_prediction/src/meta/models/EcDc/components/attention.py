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

from typing import Union, Tuple
from abc import ABC, abstractmethod
import torch
from torch import nn

from meta.data.batch import SeriesBatch


class SupportSetQueryAttention(nn.Module, ABC):
    """
    Base class for support set <--> query attention.
    """

    @abstractmethod
    def forward(
        self, query: SeriesBatch, supps: SeriesBatch
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Each query attends to each time point in each series of its support set.

        Note that we cannot use a canonical attention mechanism here because of the composition of the batches.
        The support time series are stacked along the batch dimension. Suppose a fixed support set size `x`
        batch size `bs` the batch dimension is `x * bs` i.e. a batch looks like

            [ts_11, ts_12, ...,ts_1x, ts_21, ts_22, ...ts_2x, ...,ts_bs1, ts_bs2, ..., ts_bsx].

        This is done to avoid padding and support differing support set sizes.
        However, this als means that each query should attend over multiple elements ts_*1, ...ts_*x
        along the batch dimension.

        Parameters
        ----------
        query:
            A SeriesBatch containing sequences of size [q_batch, q_size].
        supps:
             A SeriesBatch containing sequences of size [supps_batch, sequence length, supps_size].

        Returns
        -------
        torch.Tensor:
            The transformed batch containing sequences with the encoded support set series
            of size [q_batch, embedding size].
        """


class MultiHeadSupportSetQueryAttention(SupportSetQueryAttention):
    """

    Parameters
    ----------
    supps_size: The encoding dimension of the support series.
    q_size: The encoding dimension of the query series.
    embed_size: The embedding dimension for keys, queries and values.
    num_heads: The number of attention heads
    """

    def __init__(self, supps_size: int, q_size: int, num_heads: int = 1):
        super().__init__()
        assert (
            q_size % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.supps_embed_size = supps_size
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=q_size,
            kdim=self.supps_embed_size,
            vdim=self.supps_embed_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, supps: SeriesBatch, query: SeriesBatch, mask=None):
        """
        Returns
        -------
        torch.Tensor:
            Aggregated values of size [batch, embed_size]
        list:
            optional, list of attention score tensors. Each tensor has size [n_supps, 1, support ts length].
            support ts length might be different from item to item in the list.
        """
        n_supps = supps.split_sections[0]
        pad_size = supps.sequences.size()[1]
        support_set = supps.sequences.reshape(
            -1, n_supps * pad_size, self.supps_embed_size
        )
        device = supps.sequences.device
        mask = torch.arange(pad_size, device=device)[None, :] < supps.lengths[
            :, None
        ].to(device)
        mask = ~mask.reshape(-1, n_supps * pad_size)

        values, attention = self.multihead_attn(
            query=query.sequences.unsqueeze(1),
            key=support_set,
            value=support_set,
            key_padding_mask=mask,
        )
        return values.squeeze(1), attention
