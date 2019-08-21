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

# Standard library imports
from typing import List, Optional

# First-party imports
from gluonts.block.decoder import ForkingMLPDecoder
from gluonts.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    RNNEncoder,
    Seq2SeqEncoder,
)
from gluonts.block.quantile_output import QuantileOutput
from gluonts.core.component import validated
from gluonts.trainer import Trainer

# Relative imports
from ._forking_estimator import ForkingSeq2SeqEstimator


class MQDNNEstimator(ForkingSeq2SeqEstimator):
    """
    Intermediate base class for a Multi-horizon Quantile Deep Neural Network
    (MQ-DNN), [WTN+17]_. The class fixes the decoder is a multi-quantile MLP.
    Subclasses fix the encoder to be either a Convolutional Neural Network
    (MQ-CNN) or a Recurrent Neural Network (MQ-RNN).
    """

    @validated()
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        context_length: Optional[int],
        prediction_length: int,
        freq: str,
        # FIXME: why do we have two parameters here?
        mlp_final_dim: int = 20,
        mlp_hidden_dimension_seq: List[int] = list(),
        quantiles: List[float] = list(),
        trainer: Trainer = Trainer(),
    ) -> None:
        context_length = (
            prediction_length if context_length is None else context_length
        )
        assert all(
            [d > 0 for d in mlp_hidden_dimension_seq]
        ), "Elements of `mlp_hidden_dimension_seq` should be > 0"

        decoder = ForkingMLPDecoder(
            dec_len=prediction_length,
            final_dim=mlp_final_dim,
            hidden_dimension_sequence=mlp_hidden_dimension_seq,
            prefix="decoder_",
        )

        quantile_output = QuantileOutput(quantiles)

        super(MQDNNEstimator, self).__init__(
            encoder=encoder,
            decoder=decoder,
            quantile_output=quantile_output,
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
        )


class MQCNNEstimator(MQDNNEstimator):
    """
    An :class:`MQDNNEstimator` with a Convolutional Neural Network (CNN) as an
    encoder. Implements the MQ-CNN Forecaster, proposed in [WTN+17]_.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        context_length: Optional[int] = None,
        # FIXME: prefix those so clients know that these are decoder params
        mlp_final_dim: int = 20,
        mlp_hidden_dimension_seq: List[int] = list(),
        quantiles: List[float] = list(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
        trainer: Trainer = Trainer(),
    ) -> None:
        encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=[1, 3, 9],
            kernel_size_seq=([3] * len([30, 30, 30])),
            channels_seq=[30, 30, 30],
            use_residual=True,
            prefix="encoder_",
        )
        super(MQCNNEstimator, self).__init__(
            encoder=encoder,
            mlp_final_dim=mlp_final_dim,
            mlp_hidden_dimension_seq=mlp_hidden_dimension_seq,
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            context_length=context_length,
            quantiles=quantiles,
        )


class MQRNNEstimator(MQDNNEstimator):
    """
    An :class:`MQDNNEstimator` with a Recurrent Neural Network (RNN) as an
    encoder. Implements the MQ-RNN Forecaster, proposed in [WTN+17]_.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        context_length: Optional[int] = None,
        # FIXME: prefix those so clients know that these are decoder params
        mlp_final_dim: int = 20,
        mlp_hidden_dimension_seq: List[int] = list(),
        trainer: Trainer = Trainer(),
        quantiles: List[float] = list([0.1, 0.5, 0.9]),
    ) -> None:
        encoder = RNNEncoder(
            mode="gru",
            hidden_size=50,
            num_layers=1,
            bidirectional=True,
            prefix="encoder_",
        )
        super(MQRNNEstimator, self).__init__(
            encoder=encoder,
            mlp_final_dim=mlp_final_dim,
            mlp_hidden_dimension_seq=mlp_hidden_dimension_seq,
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            context_length=context_length,
            quantiles=quantiles,
        )
