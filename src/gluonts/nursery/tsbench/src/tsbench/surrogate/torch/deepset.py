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

from typing import List
import torch
from torch import nn


class DeepSetModel(nn.Module):
    """
    A model built on the DeepSet architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        encoder_dims: List[int],
        decoder_dims: List[int],
        dropout: float,
    ):
        super().__init__()

        self.encoder = _MLP([input_dim] + encoder_dims + [hidden_dim], dropout)
        self.decoder = _MLP(
            [hidden_dim + 1] + decoder_dims + [output_dim], dropout
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Computes the model output by mapping all inputs independently into the
        latent space and averaging contiguous blocks of latent representations
        as specified by the lengths.

        Then passes the averaged latent representations along with the number
        of members that have been averaged to the decoder.
        """
        # Run encoder
        z = self.encoder(x)

        # Sum encodings of different length together
        indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        indices[lengths.cumsum(0)[:-1]] = 1
        groups = indices.cumsum(0)

        encodings = torch.zeros(
            lengths.size(0), z.size(1), dtype=z.dtype, device=z.device
        )
        encodings.index_add_(0, groups, z)

        # Run decoder
        return self.decoder(
            torch.cat(
                [encodings / lengths, lengths.float().unsqueeze(1)], dim=1
            )
        )


class _MLP(nn.Sequential):
    def __init__(self, layer_dims: List[int], dropout: float):
        layers = []
        for i, (in_size, out_size) in enumerate(
            zip(layer_dims, layer_dims[1:])
        ):
            if i > 0:
                layers.append(nn.LeakyReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_size, out_size))
        super().__init__(*layers)
