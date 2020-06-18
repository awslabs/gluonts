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

# Third-party imports
from typing import List

# Third-party imports
import mxnet as mx
import numpy as np
from mxnet import gluon

from gluonts.core.component import DType, validated
from gluonts.model.common import Tensor

# First-party imports
from gluonts.mx.block.decoder import Seq2SeqDecoder
from gluonts.mx.block.enc2dec import Seq2SeqEnc2Dec
from gluonts.mx.block.encoder import Seq2SeqEncoder
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.support.util import weighted_average


class ForkingSeq2SeqNetworkBase(gluon.HybridBlock):
    """
    Base network for the :class:`ForkingSeq2SeqEstimator`.

    Parameters
    ----------
    encoder: Seq2SeqEncoder
        encoder block
    enc2dec: Seq2SeqEnc2Dec
        encoder to decoder mapping block
    decoder: Seq2SeqDecoder
        decoder block
    quantile_output: QuantileOutput
        quantile output block
    context_length: int,
        length of the encoding sequence
    cardinality: List[int],
        number of values of each categorical feature.
    embedding_dimension: List[int],
        dimension of the embeddings for categorical features
    scaling
        Whether to automatically scale the target values (default: True)
    dtype
        (default: np.float32)
    kwargs: dict
        dictionary of Gluon HybridBlock parameters
    """

    @validated()
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        enc2dec: Seq2SeqEnc2Dec,
        decoder: Seq2SeqDecoder,
        quantile_output: QuantileOutput,
        context_length: int,
        cardinality: List[int],
        embedding_dimension: List[int],
        scaling: bool = True,
        dtype: DType = np.float32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.encoder = encoder
        self.enc2dec = enc2dec
        self.decoder = decoder
        self.quantile_output = quantile_output
        self.context_length = context_length
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.dtype = dtype

        if self.scaling:
            self.scaler = MeanScaler(keepdims=True)
        else:
            self.scaler = NOPScaler(keepdims=True)

        with self.name_scope():
            self.quantile_proj = quantile_output.get_quantile_proj()
            self.loss = quantile_output.get_loss()
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=embedding_dimension,
                dtype=self.dtype,
            )

    # this method connects the sub-networks and returns the decoder output
    def get_decoder_network_output(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic: Tensor,
        future_feat_dynamic: Tensor,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
    ) -> Tensor:

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        scaled_past_target, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )

        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help prediction too
        # (batch_size, num_features + prod(target_shape))
        feat_static_real = F.concat(
            embedded_cat, F.log(scale.squeeze(axis=1)), dim=1,
        )

        # Passing past_observed_values as a feature would allow the network to
        # make that distinction and possibly ignore the masked values.
        past_feat_dynamic_extended = F.concat(
            past_feat_dynamic, past_observed_values, dim=-1
        )

        # arguments: target, static_features, dynamic_features
        enc_output_static, enc_output_dynamic = self.encoder(
            scaled_past_target, feat_static_real, past_feat_dynamic_extended
        )

        # arguments: encoder_output_static, encoder_output_dynamic, future_features
        dec_input_static, dec_input_dynamic = self.enc2dec(
            enc_output_static, enc_output_dynamic, future_feat_dynamic
        )

        # arguments: dynamic_input, static_input
        # TODO: optimize what we pass to the decoder for the prediction case,
        #  where we we only need to pass the encoder output for the last time step
        dec_output = self.decoder(dec_input_dynamic, dec_input_static)

        # the output shape should be: (batch_size, enc_len, dec_len, final_dims)
        return dec_output


class ForkingSeq2SeqTrainingNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        past_feat_dynamic: Tensor,
        future_feat_dynamic: Tensor,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            shape (batch_size, encoder_length, 1)
        future_target: Tensor
            shape (batch_size, encoder_length, decoder_length)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_feature_dynamic)
        future_feat_dynamic
            shape (batch_size, encoder_length, decoder_length, num_feature_dynamic)
        feat_static_cat
            shape (batch_size, encoder_length, num_feature_static_cat)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)
        future_observed_values: Tensor
            shape (batch_size, encoder_length, decoder_length)

        Returns
        -------
        loss with shape (batch_size, prediction_length)
        """
        dec_output = self.get_decoder_network_output(
            F,
            past_target,
            past_feat_dynamic,
            future_feat_dynamic,
            feat_static_cat,
            past_observed_values,
        )

        dec_dist_output = self.quantile_proj(dec_output)
        loss = self.loss(future_target, dec_dist_output)

        # mask the loss based on observed indicator
        weighted_loss = weighted_average(
            F=F, x=loss, weights=future_observed_values, axis=1
        )

        return weighted_loss


class ForkingSeq2SeqPredictionNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic: Tensor,
        future_feat_dynamic: Tensor,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
             shape (batch_size, encoder_length, 1)
        feat_static_cat
            shape (batch_size, encoder_length, num_feature_static_cat)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_feature_dynamic)
        future_feat_dynamic
            shape (batch_size, encoder_length, decoder_length, num_feature_dynamic)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)

        Returns
        -------
        prediction tensor with shape (batch_size, prediction_length)
        """

        dec_output = self.get_decoder_network_output(
            F,
            past_target,
            past_feat_dynamic,
            future_feat_dynamic,
            feat_static_cat,
            past_observed_values,
        )

        # We only care about the output of the decoder for the last time step
        fcst_output = F.slice_axis(dec_output, axis=1, begin=-1, end=None)
        fcst_output = F.squeeze(fcst_output, axis=1)

        predictions = self.quantile_proj(fcst_output).swapaxes(2, 1)

        return predictions
