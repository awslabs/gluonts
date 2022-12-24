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

from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module, ABC):
    """
    Base class for decoders.
    """

    @abstractmethod
    def forward(
        self, query: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Forecast from encoded (query, support set) (typically via attention) and encoded queries.

        Parameters
        ----------
        value: Encoded (query, support set).
        query: Encoded query.

        Returns
        -------
        torch.Tensor: predictions of size [n_q, *, *]
        """


class FeedForwardQuantileDecoder(Decoder):
    """
    A simple linear decoder.

    Parameters
    ----------
    embed_size: The size of the value embedding dimension.
    q_size: The size of the query encoding.
    prediction_length: The length of the prediction horizon.
    num_quantiles: The number of quantiles to be predicted.


    Returns
    -------
    torch.Tensor: predictions of size [batch, prediction_length, num_quantiles]

    """

    def __init__(
        self,
        num_quantiles: int,
        embed_size: int = 10,
        q_size: int = 20,
        hidden_size: int = 32,
        prediction_length: int = 1,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.prediction_length = prediction_length
        self.fc1 = nn.Linear(embed_size + q_size, hidden_size)
        self.fc2 = nn.Linear(
            hidden_size, prediction_length * self.num_quantiles
        )

    def forward(
        self, query: torch.Tensor, value: torch.Tensor = None
    ) -> torch.Tensor:
        if value is not None:
            x = torch.cat((value, query.sequences), dim=1)
        else:
            x = query.sequences
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(-1, self.prediction_length, self.num_quantiles)
        return x
