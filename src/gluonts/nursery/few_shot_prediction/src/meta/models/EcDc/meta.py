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
from meta.models.EcDc.components import CausalCNNEncoder
from lightkit.nn import Configurable
from dataclasses import dataclass, field
from meta.models.registry import register_model

from meta.data.batch import SeriesBatch
from meta.models.model import MetaModel
from .components import (
    FeatureExtractor,
    SupportSetEncoder,
    QueryEncoder,
    SupportSetQueryAttention,
    Decoder,
    TcnQueryEncoder,
    IdentityFeatureExtractor,
    LSTMSupportSetEncoder,
    LSTMQueryEncoder,
    FeedForwardQuantileDecoder,
    MultiHeadSupportSetQueryAttention,
    CNNSupportSetEncoder,
    TcnSupportSetEncoder,
)


class EncoderDecoderMetaModel(MetaModel):
    """
    Base class for meta models with a specific encoder - decoder architecture.

    Parameters
    ----------
    feature_extractor: A feature extractor that is applied to queries and support series alike
        and computes features for each time step.
    supps_enc: A support set encoder that encodes each time step of each support series.
    query_enc: A query encoder that encodes each query into a global presentation (not each time step).
    attention: Attention over encoded query and support series.
    decoder: Forecasts from attention(query, supps) and encoded query.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        supps_enc: SupportSetEncoder,
        query_enc: QueryEncoder,
        attention: SupportSetQueryAttention,
        decoder: Decoder,
    ):
        MetaModel.__init__(self)
        self.feature_extractor = feature_extractor
        self.supps_enc = supps_enc
        self.query_enc = query_enc
        self.attention = attention
        self.decoder = decoder

    def forward(
        self,
        supps: SeriesBatch,
        query: SeriesBatch,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Computes the forecasts from the provided queries and support set.

        Parameters
        ----------
        supps: The support set.
        query: The query.
        return_attention: If true, also return attention.
        """
        supps = self.feature_extractor(supps)
        query = self.feature_extractor(query)

        supps = self.supps_enc(supps)
        query = self.query_enc(query)

        value, attention = self.attention(query=query, supps=supps)
        result = self.decoder(query=query, value=value)
        if return_attention:
            return result, attention
        else:
            return result


@dataclass
class IwataKumagaiEcDcConfig:
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
    query_out_channels: int = 32
    lstm_num_layers: int = 1
    supps_out_channels: int = 32
    supps_bidirectional: bool = True
    num_heads: int = 1
    decoder_hidden_size: int = 32


@register_model
class IwataKumagaiEcDc(
    EncoderDecoderMetaModel, Configurable[IwataKumagaiEcDcConfig]
):
    """
    The base model from the paper https://arxiv.org/abs/2009.14379 by Iwata and Kumagai.
    Differences are quantile prediction and (optionally) longer prediction window.

    """

    def __init__(self, config: IwataKumagaiEcDcConfig):
        Configurable.__init__(self, config)
        EncoderDecoderMetaModel.__init__(
            self,
            feature_extractor=IdentityFeatureExtractor(),
            supps_enc=LSTMSupportSetEncoder(
                input_size=1,
                hidden_size=config.supps_out_channels,
                num_layers=config.lstm_num_layers,
                bidirectional=config.supps_bidirectional,
            ),
            query_enc=LSTMQueryEncoder(
                input_size=1,
                num_layers=config.lstm_num_layers,
                hidden_size=config.query_out_channels,
            ),
            attention=MultiHeadSupportSetQueryAttention(
                supps_size=config.supps_out_channels
                * (1 + config.supps_bidirectional),
                q_size=config.query_out_channels,
                num_heads=config.num_heads,
            ),
            decoder=FeedForwardQuantileDecoder(
                embed_size=config.query_out_channels,
                q_size=config.query_out_channels,
                hidden_size=config.decoder_hidden_size,
                prediction_length=config.prediction_length,
                num_quantiles=len(config.quantiles),
            ),
        )

    @classmethod
    def name(cls) -> str:
        return "iwata"


@dataclass
class CNNLSTMEcDcConfig:
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
    supps_out_channels: int = 64
    num_heads: int = 1
    attention_embed_size: int = 32


@register_model
class CNNLSTMEcDc(EncoderDecoderMetaModel, Configurable[CNNLSTMEcDcConfig]):
    """
    Support set is encoded by a CNN, query by a LSTM.
    """

    def __init__(self, config: CNNLSTMEcDcConfig):
        Configurable.__init__(self, config)
        EncoderDecoderMetaModel.__init__(
            self,
            feature_extractor=IdentityFeatureExtractor(),
            supps_enc=CNNSupportSetEncoder(
                input_size=1, out_channels=config.supps_out_channels
            ),
            query_enc=LSTMQueryEncoder(
                input_size=1,
                num_layers=1,
                hidden_size=32,
            ),
            attention=MultiHeadSupportSetQueryAttention(
                supps_size=config.supps_out_channels,
                q_size=32,
                num_heads=config.num_heads,
            ),
            decoder=FeedForwardQuantileDecoder(
                embed_size=config.attention_embed_size,
                q_size=32,
                prediction_length=config.prediction_length,
                num_quantiles=len(config.quantiles),
            ),
        )

    @classmethod
    def name(cls) -> str:
        return "cnn_iwata"


@dataclass
class TcnEcDcConfig:
    """
    Configuration class for a  TcnEcDc.
    See also:
        :class:` TcnEcDc`
    """

    # -------- decoder config ----------
    prediction_length: int = 1
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
    decoder_hidden_size: int = 64

    # -------- encoder config ----------
    num_channels: int = 64
    kernel_size: int = 2
    num_layers: int = 5

    # -------- attention config ----------
    num_heads: int = 1


@register_model
class TcnEcDc(EncoderDecoderMetaModel, Configurable[TcnEcDcConfig]):
    """
    Shared WaveNet-like encoder for query and support set.
    Multi-head attention to match the encoded query and support set.
    Feedforward decoder for multi-step quantile prediction.
    """

    def __init__(self, config: TcnEcDcConfig):
        common_encoder = CausalCNNEncoder(
            in_channels=1,
            channels=config.num_channels,
            depth=config.num_layers,
            out_channels=config.num_channels,
            kernel_size=config.kernel_size,
        )
        Configurable.__init__(self, config)
        EncoderDecoderMetaModel.__init__(
            self,
            feature_extractor=IdentityFeatureExtractor(),
            supps_enc=TcnSupportSetEncoder(encoder=common_encoder),
            query_enc=TcnQueryEncoder(
                encoder=common_encoder,
            ),
            attention=MultiHeadSupportSetQueryAttention(
                supps_size=config.num_channels,
                q_size=config.num_channels,
                num_heads=config.num_heads,
            ),
            decoder=FeedForwardQuantileDecoder(
                embed_size=config.num_channels,
                q_size=config.num_channels,
                hidden_size=config.decoder_hidden_size,
                prediction_length=config.prediction_length,
                num_quantiles=len(config.quantiles),
            ),
        )

    @classmethod
    def name(cls) -> str:
        return "tcn"
