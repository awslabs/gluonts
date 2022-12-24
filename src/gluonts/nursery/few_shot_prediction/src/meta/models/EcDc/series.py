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
from lightkit.nn import Configurable
from dataclasses import dataclass, field
from meta.models.registry import register_model

from meta.data.batch import SeriesBatch
from meta.models.model import SeriesModel

from .components import (
    IdentityFeatureExtractor,
    LSTMQueryEncoder,
    FeedForwardQuantileDecoder,
)


@dataclass
class LSTMEncoderFeedforwardDecoderConfig:
    """
    Configuration class for a LSTMEncoderFeedforwardDecoder.
    See also:
        :class:`LSTMEncoderFeedforwardDecoder`
    """

    # The length of the prediction horizon
    prediction_length: int = 1
    # The quantiles the model predicts
    quantiles: List[str] = field(
        default_factory=lambda: [
            "0.02",
            "0.1",
            "0.25",
            "0.5",
            "0.75",
            "0.9",
            "0.98",
        ]
    )
    query_out_channels: int = 64
    # The number of layers in the encoding LSTM
    num_lstm_layers: int = 1
    decoder_hidden_size: int = 32


@register_model
class LSTMEncoderFeedforwardDecoder(
    SeriesModel, Configurable[LSTMEncoderFeedforwardDecoderConfig]
):
    """
    A simple base model that does not make use of a support set.
    """

    def __init__(self, config: LSTMEncoderFeedforwardDecoderConfig):
        Configurable.__init__(self, config)
        SeriesModel.__init__(self)
        self.feature_extractor = IdentityFeatureExtractor()
        self.query_enc = LSTMQueryEncoder(
            input_size=1,
            num_layers=config.num_lstm_layers,
            hidden_size=config.query_out_channels,
        )
        self.decoder = FeedForwardQuantileDecoder(
            embed_size=0,
            q_size=config.query_out_channels,
            hidden_size=config.decoder_hidden_size,
            prediction_length=config.prediction_length,
            num_quantiles=len(config.quantiles),
        )

    @classmethod
    def name(cls) -> str:
        return "lstm_feedforward"

    def forward(self, supps: SeriesBatch, query: SeriesBatch) -> torch.Tensor:
        """
        Computes the forecasts from the provided queries.

        Parameters
        ----------
        supps: Not used!
        query: The query.
        """

        x = self.feature_extractor(query)
        x = self.query_enc(query=x)
        x = self.decoder(query=x)
        return x
