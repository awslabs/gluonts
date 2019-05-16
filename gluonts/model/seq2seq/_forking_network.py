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

        with self.name_scope():
            self.quantile_proj = quantile_output.get_quantile_proj()
            self.loss = quantile_output.get_loss()


class ForkingSeq2SeqTrainingNetwork(ForkingSeq2SeqNetworkBase):
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

        # FIXME: can we factor out a common prefix in the base network?
        feat_static_real = nd_None
        past_feat_dynamic_real = nd_None
        future_feat_dynamic_real = nd_None

        enc_output_static, enc_output_dynamic = self.encoder(
            past_target, feat_static_real, past_feat_dynamic_real
        )

        dec_input_static, dec_input_dynamic, _ = self.enc2dec(
            enc_output_static, enc_output_dynamic, future_feat_dynamic_real
        )

        dec_output = self.decoder(dec_input_dynamic, dec_input_static)
        dec_dist_output = self.quantile_proj(dec_output)

        loss = self.loss(future_target, dec_dist_output)
        return loss.mean(axis=1)


class ForkingSeq2SeqPredictionNetwork(ForkingSeq2SeqNetworkBase):
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
        feat_static_real = nd_None
        past_feat_dynamic_real = nd_None
        future_feat_dynamic_real = nd_None

        enc_output_static, enc_output_dynamic = self.encoder(
            past_target, feat_static_real, past_feat_dynamic_real
        )

        enc_output_static = (
            nd_None if enc_output_static is None else enc_output_static
        )

        dec_inp_static, dec_inp_dynamic, _ = self.enc2dec(
            enc_output_static, enc_output_dynamic, future_feat_dynamic_real
        )

        dec_output = self.decoder(dec_inp_dynamic, dec_inp_static)
        fcst_output = F.slice_axis(dec_output, axis=1, begin=-1, end=None)
        fcst_output = F.squeeze(fcst_output, axis=1)

        predictions = self.quantile_proj(fcst_output).swapaxes(2, 1)
        return predictions
