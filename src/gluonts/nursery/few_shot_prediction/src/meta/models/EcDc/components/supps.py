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
from meta.models.EcDc.components.tcn import CausalCNNEncoder
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from meta.data.batch import SeriesBatch


class SupportSetEncoder(nn.Module, ABC):
    """
    Base class for support set encoders.
    """

    @abstractmethod
    def forward(self, supps: SeriesBatch) -> SeriesBatch:
        """
        Encodes each time point of each support set time series with a vector of fixed size.

        Parameters
        ----------
        supps:
            A SeriesBatch containing sequences of size [batch, sequence length, n_features].

        Returns
        -------
        SeriesBatch:
            The transformed batch containing sequences with the encoded support set series
            of size [batch, sequence length, encoding size].
        """


class LSTMSupportSetEncoder(SupportSetEncoder):
    """
    Encodes each time step in the support set times series via the hidden states of a LSTM.

    Parameters
    ----------
    input_size: The number of features of the time series.
    hidden_size: The size of hidden states of LSTM, corresponds to encoding size of the queries,
        i.e. the encoder returns a tensor with shape [batch, sequence length, D * hidden_size].
    num_layers: number of layers of the LSTM.
    bidirectional: If a bidirectional LSTM is used. If true: D = 2, else D = 1.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, supps: SeriesBatch) -> torch.Tensor:
        supps_packed = pack_padded_sequence(
            supps.sequences,
            supps.lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output_packed, _ = self.encoder(supps_packed)
        output_padded, output_lengths = pad_packed_sequence(
            output_packed, batch_first=True
        )
        return SeriesBatch(output_padded, output_lengths, supps.split_sections)


class CNNSupportSetEncoder(SupportSetEncoder):
    """
    Encodes the time series in the support set via a CNN.

    Parameters
    ----------
    input_size: The number of features of the time series.
    out_channels: The number of output channels of the CNN, corresponds to encoding size of the queries,
        i.e. the encoder returns a tensor with shape [batch, sequence length, D * hidden_size].
    """

    def __init__(self, input_size: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=out_channels,
            kernel_size=5,
            padding="valid",
        )

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding="same",
        )
        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.conv4 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )

    def forward(self, supps: SeriesBatch) -> torch.Tensor:
        # TODO: permute should be handled as here:
        # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        # https://discuss.pytorch.org/t/dataset-with-last-dimension-as-channel/100639/6
        pool = F.avg_pool1d
        channels_first = supps.sequences.permute(0, 2, 1)
        x = pool(F.relu(self.conv1(channels_first)), 2)
        x = pool(F.relu(self.conv2(x)), 2)
        x = pool(F.relu(self.conv3(x)), 2)
        x = pool(F.relu(self.conv4(x)), 2)

        # TODO: padding of the batches is not takes into account!
        return SeriesBatch(
            x.permute(0, 2, 1), supps.lengths, supps.split_sections
        )


class TcnSupportSetEncoder(SupportSetEncoder):
    """
    Encodes each time point of the support set time series via WaveNet

    Parameters
    ----------
    num_channels: The number of filters in each CNN layer.
    num_layers: The number of causal convolution blocks
    kernel_size: The kernel size of the filters in the causal convolution blocks
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

    def forward(self, supps: SeriesBatch) -> torch.Tensor:
        output_padded = self.encoder(supps.sequences.transpose(1, 2))
        return SeriesBatch(
            output_padded.transpose(1, 2), supps.lengths, supps.split_sections
        )
