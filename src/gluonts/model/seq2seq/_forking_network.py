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

from typing import List, Optional, Tuple

import mxnet as mx
import numpy as np
from mxnet import gluon

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor
from gluonts.mx.block.decoder import Seq2SeqDecoder
from gluonts.mx.block.enc2dec import Seq2SeqEnc2Dec
from gluonts.mx.block.encoder import Seq2SeqEncoder
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.util import weighted_average


class ForkingSeq2SeqNetworkBase(gluon.HybridBlock):
    """
    Base network for the :class:`ForkingSeq2SeqEstimator`.

    Parameters
    ----------
    encoder: Seq2SeqEncoder
        encoder block.
    enc2dec: Seq2SeqEnc2Dec
        encoder to decoder mapping block.
    decoder: Seq2SeqDecoder
        decoder block.
    quantile_output
        quantile output
    distr_output
        distribution output
    context_length: int,
        length of the encoding sequence.
    cardinality: List[int],
        number of values of each categorical feature.
    embedding_dimension: List[int],
        dimension of the embeddings for categorical features.
    scaling
        Whether to automatically scale the target values. (default: True)
    scaling_decoder_dynamic_feature
        Whether to automatically scale the dynamic features for the decoder. (default: False)
    dtype
        (default: np.float32)
    num_forking: int,
        decides how much forking to do in the decoder. 1 reduces to seq2seq and enc_len reduces to MQ-C(R)NN.
    kwargs: dict
        dictionary of Gluon HybridBlock parameters
    """

    @validated()
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        enc2dec: Seq2SeqEnc2Dec,
        decoder: Seq2SeqDecoder,
        context_length: int,
        cardinality: List[int],
        embedding_dimension: List[int],
        distr_output: Optional[DistributionOutput] = None,
        quantile_output: Optional[QuantileOutput] = None,
        scaling: bool = True,
        scaling_decoder_dynamic_feature: bool = False,
        dtype: DType = np.float32,
        num_forking: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert (distr_output is None) != (quantile_output is None)

        self.encoder = encoder
        self.enc2dec = enc2dec
        self.decoder = decoder
        self.distr_output = distr_output
        self.quantile_output = quantile_output
        self.scaling = scaling
        self.scaling_decoder_dynamic_feature = scaling_decoder_dynamic_feature
        self.dtype = dtype
        self.num_forking = (
            num_forking if num_forking is not None else context_length
        )

        if self.scaling:
            self.scaler = MeanScaler()
        else:
            self.scaler = NOPScaler()

        if self.scaling_decoder_dynamic_feature:
            self.scaler_decoder_dynamic_feature = MeanScaler(axis=1)
        else:
            self.scaler_decoder_dynamic_feature = NOPScaler(axis=1)

        with self.name_scope():
            if self.quantile_output:
                self.quantile_proj = self.quantile_output.get_quantile_proj()
                self.loss = self.quantile_output.get_loss()
            else:
                assert self.distr_output is not None
                self.distr_args_proj = self.distr_output.get_args_proj()
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
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            shape (batch_size, encoder_length, 1)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_past_feat_dynamic)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, num_feat_dynamic)
        feat_static_cat
            shape (batch_size, num_feat_static_cat)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)
        Returns
        -------
        decoder output tensor of size (batch_size, num_forking, dec_len, decoder_mlp_dim_seq[0])
        """

        # scale shape: (batch_size, 1, 1)
        scaled_past_target, scale = self.scaler(
            past_target, past_observed_values
        )

        # (batch_size, sum(embedding_dimension) = num_feat_static_cat)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help prediction too
        # (batch_size, num_feat_static = sum(embedding_dimension) + 1)
        feat_static_real = F.concat(embedded_cat, F.log(scale), dim=1)

        # Passing past_observed_values as a feature would allow the network to
        # make that distinction and possibly ignore the masked values.
        past_feat_dynamic_extended = F.concat(
            past_feat_dynamic, past_observed_values, dim=-1
        )

        # arguments: target, static_features, dynamic_features
        # enc_output_static shape: (batch_size, channels_seq[-1] + 1)
        # enc_output_dynamic shape: (batch_size, encoder_length, channels_seq[-1] + 1)
        enc_output_static, enc_output_dynamic = self.encoder(
            scaled_past_target, feat_static_real, past_feat_dynamic_extended
        )

        # TODO: This assumes that future_feat_dynamic has no missing values
        # TODO: Output the scale as well to be used by the decoder
        scaled_future_feat_dynamic, _ = self.scaler_decoder_dynamic_feature(
            future_feat_dynamic, F.ones_like(future_feat_dynamic)
        )

        # arguments: encoder_output_static, encoder_output_dynamic, future_features
        # dec_input_static shape: (batch_size, channels_seq[-1] + 1)
        # dec_input_dynamic shape:(batch_size, num_forking, channels_seq[-1] + 1 + decoder_length * num_feat_dynamic)
        dec_input_static, dec_input_dynamic = self.enc2dec(
            enc_output_static,
            # slice axis 1 from encoder_length = context_length to num_forking
            enc_output_dynamic.slice_axis(
                axis=1, begin=-self.num_forking, end=None
            ),
            scaled_future_feat_dynamic,
        )

        # arguments: dynamic_input, static_input
        # TODO: optimize what we pass to the decoder for the prediction case,
        #  where we we only need to pass the encoder output for the last time step
        dec_output = self.decoder(dec_input_static, dec_input_dynamic)

        # the output shape should be: (batch_size, num_forking, dec_len, decoder_mlp_dim_seq[0])
        return dec_output, scale


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
            shape (batch_size, num_forking, decoder_length)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_past_feat_dynamic)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, num_feat_dynamic)
        feat_static_cat
            shape (batch_size, num_feat_static_cat)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)
        future_observed_values: Tensor
            shape (batch_size, num_forking, decoder_length)

        Returns
        -------
        loss with shape (batch_size, prediction_length)
        """
        # shape: (batch_size, num_forking, decoder_length, decoder_mlp_dim_seq[0])
        dec_output, scale = self.get_decoder_network_output(
            F,
            past_target,
            past_feat_dynamic,
            future_feat_dynamic,
            feat_static_cat,
            past_observed_values,
        )

        if self.quantile_output is not None:
            # shape: (batch_size, num_forking, decoder_length, len(quantiles))
            dec_dist_output = self.quantile_proj(dec_output)
            # shape: (batch_size, num_forking, decoder_length = prediction_length)
            loss = self.loss(future_target, dec_dist_output)
        else:
            assert self.distr_output is not None
            distr_args = self.distr_args_proj(dec_output)
            distr = self.distr_output.distribution(
                distr_args, scale=scale.expand_dims(axis=1)
            )
            loss = distr.loss(future_target)

        # mask the loss based on observed indicator
        # shape: (batch_size, decoder_length)
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
            shape (batch_size, num_feat_static_cat)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_past_feat_dynamic)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, num_feat_dynamic)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)

        Returns
        -------
        prediction tensor with shape (batch_size, prediction_length)
        """

        # shape: (batch_size, num_forking, decoder_length, decoder_mlp_dim_seq[0])
        dec_output, _ = self.get_decoder_network_output(
            F,
            past_target,
            past_feat_dynamic,
            future_feat_dynamic,
            feat_static_cat,
            past_observed_values,
        )

        # We only care about the output of the decoder for the last time step
        # shape: (batch_size, decoder_length, decoder_mlp_dim_seq[0])
        fcst_output = F.slice_axis(dec_output, axis=1, begin=-1, end=None)
        fcst_output = F.squeeze(fcst_output, axis=1)

        # shape: (batch_size, len(quantiles), decoder_length = prediction_length)
        predictions = self.quantile_proj(fcst_output).swapaxes(2, 1)

        return predictions


class ForkingSeq2SeqDistributionPredictionNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic: Tensor,
        future_feat_dynamic: Tensor,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
            shape (batch_size, num_forking, decoder_length, num_feature_dynamic)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)
        Returns
        -------
        distr_args: the parameters of distribution
        loc: an array of zeros with the same shape of scale
        scale:
        """

        dec_output, scale = self.get_decoder_network_output(
            F,
            past_target,
            past_feat_dynamic,
            future_feat_dynamic,
            feat_static_cat,
            past_observed_values,
        )
        fcst_output = F.slice_axis(dec_output, axis=1, begin=-1, end=None)
        fcst_output = F.squeeze(fcst_output, axis=1)
        distr_args = self.distr_args_proj(fcst_output)

        loc = F.zeros_like(scale)
        return distr_args, loc, scale
