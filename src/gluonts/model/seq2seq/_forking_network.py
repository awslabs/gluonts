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
from mxnet import gluon, nd

# First-party imports
from gluonts.block.decoder import Seq2SeqDecoder
from gluonts.block.enc2dec import Seq2SeqEnc2Dec
from gluonts.block.encoder import Seq2SeqEncoder
from gluonts.block.quantile_output import QuantileOutput
from gluonts.core.component import validated
from gluonts.model.common import Tensor

nd_None = nd.array([])


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

        self.feat_static_real = nd_None
        self.past_feat_dynamic_real = nd_None
        self.future_feat_dynamic_real = nd_None

        with self.name_scope():
            self.quantile_proj = quantile_output.get_quantile_proj()
            self.loss = quantile_output.get_loss()


class ForkingSeq2SeqNetwork:
    @validated()
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        enc2dec: Seq2SeqEnc2Dec,
        decoder: Seq2SeqDecoder,
        quantile_output: QuantileOutput,
        use_dynamic_real: bool = False,
        use_static_cat: bool = False,
        **kwargs,
    ) -> None:
        self.encoder = encoder
        self.enc2dec = enc2dec
        self.decoder = decoder
        self.quantile_output = quantile_output

        self.use_dynamic_real = use_dynamic_real
        self.use_static_cat = use_static_cat

    def get_training_network(self) -> ForkingSeq2SeqNetworkBase:
        if self.use_static_cat is False and self.use_dynamic_real is False:
            return ForkingSeq2SeqTargetTrainingNetwork(
                encoder=self.encoder,
                enc2dec=self.enc2dec,
                decoder=self.decoder,
                quantile_output=self.quantile_output,
            )
        elif self.use_static_cat is False and self.use_dynamic_real:
            return ForkingSeq2SeqTrainingNetwork(
                encoder=self.encoder,
                enc2dec=self.enc2dec,
                decoder=self.decoder,
                quantile_output=self.quantile_output,
            )
        else:
            raise NotImplementedError

    def get_prediction_network(self) -> ForkingSeq2SeqNetworkBase:
        if self.use_static_cat is False and self.use_dynamic_real is False:
            return ForkingSeq2SeqTargetPredictionNetwork(
                encoder=self.encoder,
                enc2dec=self.enc2dec,
                decoder=self.decoder,
                quantile_output=self.quantile_output,
            )
        elif self.use_static_cat is False and self.use_dynamic_real:
            return ForkingSeq2SeqPredictionNetwork(
                encoder=self.encoder,
                enc2dec=self.enc2dec,
                decoder=self.decoder,
                quantile_output=self.quantile_output,
            )
        else:
            raise NotImplementedError


# TODO: figure out whether we need 2 classes each, in fact we would need 4 each,
#  if adding categorical with this technique, does not seem reasonable
class ForkingSeq2SeqTrainingNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Tensor,
        future_target: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            FIXME
        future_target: Tensor
            shape (num_ts, encoder_length, 1) FIXME

        Returns
        -------
        loss with shape (FIXME, FIXME)
        """

        # print(f"past target: {past_target.shape}")
        # print(f"past_feat_dynamic_real: {past_feat_dynamic_real.shape}")
        # print(f"future_target: {future_target.shape}")

        # arguments: target, static_features, dynamic_features
        enc_output_static, enc_output_dynamic = self.encoder(
            past_target, self.feat_static_real, past_feat_dynamic_real
        )

        # arguments: encoder_output_static, encoder_output_dynamic, future_features
        # TODO: figure out how future_features is supposed to be used: since no distinction
        #  between dynamic and static anymore (shape is (N, T, C) suggesting dynamic feature)
        dec_input_static, dec_input_dynamic, _ = self.enc2dec(
            enc_output_static,
            enc_output_dynamic,
            self.future_feat_dynamic_real,
        )

        dec_output = self.decoder(dec_input_dynamic, dec_input_static)
        dec_dist_output = self.quantile_proj(dec_output)

        # print(f"decoder output: {dec_dist_output.shape}")

        loss = self.loss(future_target, dec_dist_output)
        return loss.mean(axis=1)


class ForkingSeq2SeqPredictionNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, past_target: Tensor, past_feat_dynamic_real: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            FIXME

        Returns
        -------
        prediction tensor with shape (FIXME, FIXME)
        """

        # FIXME: can we factor out a common prefix in the base network?

        enc_output_static, enc_output_dynamic = self.encoder(
            past_target, self.feat_static_real, past_feat_dynamic_real
        )

        enc_output_static = (
            nd_None if enc_output_static is None else enc_output_static
        )

        dec_inp_static, dec_inp_dynamic, _ = self.enc2dec(
            enc_output_static,
            enc_output_dynamic,
            self.future_feat_dynamic_real,
        )

        dec_output = self.decoder(dec_inp_dynamic, dec_inp_static)
        fcst_output = F.slice_axis(dec_output, axis=1, begin=-1, end=None)
        fcst_output = F.squeeze(fcst_output, axis=1)

        predictions = self.quantile_proj(fcst_output).swapaxes(2, 1)
        return predictions


class ForkingSeq2SeqTargetTrainingNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, past_target: Tensor, future_target: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            FIXME
        future_target: Tensor
            shape (num_ts, encoder_length, 1) FIXME

        Returns
        -------
        loss with shape (FIXME, FIXME)
        """

        enc_output_static, enc_output_dynamic = self.encoder(
            past_target, self.feat_static_real, self.past_feat_dynamic_real
        )

        dec_input_static, dec_input_dynamic, _ = self.enc2dec(
            enc_output_static,
            enc_output_dynamic,
            self.future_feat_dynamic_real,
        )

        dec_output = self.decoder(dec_input_dynamic, dec_input_static)
        dec_dist_output = self.quantile_proj(dec_output)

        loss = self.loss(future_target, dec_dist_output)
        return loss.mean(axis=1)


class ForkingSeq2SeqTargetPredictionNetwork(ForkingSeq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, past_target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            FIXME

        Returns
        -------
        prediction tensor with shape (FIXME, FIXME)
        """

        # FIXME: can we factor out a common prefix in the base network?

        enc_output_static, enc_output_dynamic = self.encoder(
            past_target, self.feat_static_real, self.past_feat_dynamic_real
        )

        enc_output_static = (
            nd_None if enc_output_static is None else enc_output_static
        )

        dec_inp_static, dec_inp_dynamic, _ = self.enc2dec(
            enc_output_static,
            enc_output_dynamic,
            self.future_feat_dynamic_real,
        )

        dec_output = self.decoder(dec_inp_dynamic, dec_inp_static)
        fcst_output = F.slice_axis(dec_output, axis=1, begin=-1, end=None)
        fcst_output = F.squeeze(fcst_output, axis=1)

        predictions = self.quantile_proj(fcst_output).swapaxes(2, 1)
        return predictions
