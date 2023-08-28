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
from torch.nn.utils.rnn import pack_padded_sequence

from meta.data.batch import SeriesBatch
from .tcn import CausalCNNEncoder


class QueryEncoder(nn.Module, ABC):
    """
    Base class for query encoders.
    """

    @abstractmethod
    def forward(self, query: SeriesBatch) -> SeriesBatch:
        """
        Encodes each query with a vector of fixed size.

        Parameters
        ----------
        query:
            A SeriesBatch containing sequences of size [batch, sequence length, n_features].

        Returns
        -------
        SeriesBatch:
            The transformed batch containing sequences with the encoded queries
            of size [batch, encoding size].
        """


class LSTMQueryEncoder(QueryEncoder):
    """
    Encodes queries via the last hidden state of a uni-directional LSTM.

    Parameters
    ----------
    input_size: The number of features of the time series.
    hidden_size: The size of hidden states of LSTM, corresponds to encoding size of the queries,
        i.e. the encoder returns a tensor with shape [batch, hidden_size].
    num_layers: number of layers of the LSTM.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, query: SeriesBatch) -> torch.Tensor:
        query_packed = pack_padded_sequence(
            query.sequences,
            query.lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (output, _) = self.encoder(query_packed)
        # return last layer of hidden states
        return SeriesBatch(output[-1, ...], None, query.split_sections)


class TcnQueryEncoder(QueryEncoder):
    """
    Encodes queries via the last output of a WaveNet.

    Parameters
    ----------
    input_size: The number of features of the time series.
    hidden_size: The size of hidden states of LSTM, corresponds to encoding size of the queries,
        i.e. the encoder returns a tensor with shape [batch, hidden_size].
    num_layers: number of layers of the LSTM.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        num_channels: int = 64,
        num_layers: int = 5,
        kernel_size: int = 2,
    ):
        super().__init__()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = CausalCNNEncoder(
                in_channels=1,
                channels=num_channels,
                depth=num_layers,
                out_channels=num_channels,
                kernel_size=kernel_size,
            )

    def forward(self, query: SeriesBatch) -> torch.Tensor:
        output = self.encoder(query.sequences.permute(0, 2, 1))

        # return the last state
        idx = query.lengths - 1
        return SeriesBatch(
            output[torch.range(0, len(idx) - 1, dtype=torch.long), :, idx],
            None,
            query.split_sections,
        )
