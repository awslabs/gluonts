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
import mxnet as mx
from mxnet import gluon

# First-party imports
from gluonts.block.decoder import Seq2SeqDecoder
from gluonts.block.enc2dec import Seq2SeqEnc2Dec
from gluonts.block.encoder import Seq2SeqEncoder
from gluonts.block.quantile_output import QuantileOutput
from gluonts.core.component import validated
from gluonts.model.common import Tensor
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.encoder = encoder
        self.enc2dec = enc2dec
        self.decoder = decoder
        self.quantile_output = quantile_output

        with self.name_scope():
            self.quantile_proj = quantile_output.get_quantile_proj()
            self.loss = quantile_output.get_loss()


class ForkingSeq2SeqTrainingNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        past_observed_values: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            shape (num_ts, encoder_length, 1)
        future_target: Tensor
            shape (num_ts, encoder_length, decoder_length)
        past_observed_values: Tensor
            shape (num_ts, encoder_length, 1)
        future_observed_values: Tensor
            shape (num_ts, encoder_length, decoder_length)

        Returns
        -------
        loss with shape (num_ts, decoder_length)
        """

        # FIXME: can we factor out a common prefix in the base network?

        feat_static_real = F.zeros(shape=(1,))
        past_feat_dynamic_real = past_observed_values
        future_feat_dynamic_real = F.zeros(shape=(1,))

        enc_output_static, enc_output_dynamic = self.encoder(
            past_target, feat_static_real, past_feat_dynamic_real
        )

        dec_input_static, dec_input_dynamic, _ = self.enc2dec(
            enc_output_static, enc_output_dynamic, future_feat_dynamic_real
        )

        dec_output = self.decoder(dec_input_dynamic, dec_input_static)
        dec_dist_output = self.quantile_proj(dec_output)

        loss = self.loss(future_target, dec_dist_output)
        weighted_loss = weighted_average(
            F=F, x=loss, weights=future_observed_values, axis=1
        )
        return weighted_loss


class ForkingSeq2SeqPredictionNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, past_target: Tensor, past_observed_values: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            shape (num_ts, encoder_length, 1)
        past_observed_values: Tensor
            shape (num_ts, encoder_length, 1)

        Returns
        -------
        prediction tensor with shape (FIXME, FIXME)
        """

        # FIXME: can we factor out a common prefix in the base network?
        feat_static_real = F.zeros(shape=(1,))
        past_feat_dynamic_real = past_observed_values
        future_feat_dynamic_real = F.zeros(shape=(1,))

        enc_output_static, enc_output_dynamic = self.encoder(
            past_target, feat_static_real, past_feat_dynamic_real
        )

        enc_output_static = (
            F.zeros(shape=(1,))
            if enc_output_static is None
            else enc_output_static
        )

        dec_inp_static, dec_inp_dynamic, _ = self.enc2dec(
            enc_output_static, enc_output_dynamic, future_feat_dynamic_real
        )

        dec_output = self.decoder(dec_inp_dynamic, dec_inp_static)
        fcst_output = F.slice_axis(dec_output, axis=1, begin=-1, end=None)
        fcst_output = F.squeeze(fcst_output, axis=1)

        predictions = self.quantile_proj(fcst_output).swapaxes(2, 1)
        return predictions
