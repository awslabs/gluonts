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
from gluonts.evaluation.backtest import make_evaluation_predictions
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
from gluonts.model.seq2seq._forking_estimator import ForkingSeq2SeqEstimator
import numpy as np
import mxnet as mx


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
        decoder_mlp_dim_seq: List[int] = [20],
        quantiles: List[float] = list(),
        trainer: Trainer = Trainer(),
    ) -> None:
        context_length = (
            prediction_length if context_length is None else context_length
        )
        assert all(
            [d > 0 for d in decoder_mlp_dim_seq]
        ), "Elements of `mlp_hidden_dimension_seq` should be > 0"

        decoder = ForkingMLPDecoder(
            dec_len=prediction_length,
            final_dim=decoder_mlp_dim_seq[-1],
            hidden_dimension_sequence=decoder_mlp_dim_seq[:-1],
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
        seed: Optional[int] = None,
        decoder_mlp_dim_seq: List[int] = [20],
        channels_seq: List[int] = [30, 30, 30],
        dilation_seq: List[int] = [1, 3, 9],
        kernel_size_seq: List[int] = [3, 3, 3],
        use_residual: bool = True,
        quantiles: List[float] = list(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
        trainer: Trainer = Trainer(),
    ) -> None:

        if seed:
            np.random.seed(seed)
            mx.random.seed(seed)

        assert (
            len(channels_seq) == len(dilation_seq) == len(kernel_size_seq)
        ), (
            f"mismatch CNN configurations: {len(channels_seq)} vs. "
            f"{len(dilation_seq)} vs. {len(kernel_size_seq)}"
        )

        encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=dilation_seq,
            kernel_size_seq=channels_seq,
            channels_seq=kernel_size_seq,
            use_residual=use_residual,
            use_dynamic_feat=True,
            prefix="encoder_",
        )
        super(MQCNNEstimator, self).__init__(
            encoder=encoder,
            decoder_mlp_dim_seq=decoder_mlp_dim_seq,
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
        decoder_mlp_dim_seq: List[int] = [20],
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
            decoder_mlp_dim_seq=decoder_mlp_dim_seq,
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            context_length=context_length,
            quantiles=quantiles,
        )
