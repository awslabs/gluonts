# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.block.decoder import Seq2SeqDecoder
from gluonts.block.enc2dec import Seq2SeqEnc2Dec
from gluonts.block.encoder import Seq2SeqEncoder
from gluonts.block.feature import FeatureEmbedder
from gluonts.block.quantile_output import QuantileOutput
from gluonts.block.scaler import Scaler
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class Seq2SeqNetworkBase(mx.gluon.HybridBlock):
    """
    Base network for the :class:`Seq2SeqEstimator`.

    Parameters
    ----------
    scaler : Scaler
        scale of the target time series, both as input or in the output
        distributions
    encoder : encoder
        see encoder.py for possible choices
    enc2dec : encoder to decoder
        see enc2dec.py for possible choices
    decoder : decoder
        see decoder.py for possible choices
    quantile_output : QuantileOutput
        quantile regression output
    kwargs : dict
        a dict of parameters to be passed to the parent initializer
    """

    @validated()
    def __init__(
        self,
        embedder: FeatureEmbedder,
        scaler: Scaler,
        encoder: Seq2SeqEncoder,
        enc2dec: Seq2SeqEnc2Dec,
        decoder: Seq2SeqDecoder,
        quantile_output: QuantileOutput,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.embedder = embedder
        self.scaler = scaler
        self.encoder = encoder
        self.enc2dec = enc2dec
        self.decoder = decoder
        self.quantile_output = quantile_output

        with self.name_scope():
            self.quantile_proj = quantile_output.get_quantile_proj()
            self.loss = quantile_output.get_loss()

    def compute_decoder_outputs(
        self,
        F,
        past_target: Tensor,
        feat_static_cat: Tensor,
        past_feat_dynamic_real: Tensor,
        future_feat_dynamic_real: Tensor,
    ) -> Tensor:
        scaled_target, scale = self.scaler(
            past_target, F.ones_like(past_target)
        )

        embedded_cat = self.embedder(
            feat_static_cat
        )  # (batch_size, num_features * embedding_size)

        encoder_output_static, encoder_output_dynamic = self.encoder(
            scaled_target, embedded_cat, past_feat_dynamic_real
        )
        decoder_input_static, _, decoder_input_dynamic = self.enc2dec(
            encoder_output_static,
            encoder_output_dynamic,
            future_feat_dynamic_real,
        )
        decoder_output = self.decoder(
            decoder_input_static, decoder_input_dynamic
        )
        scaled_decoder_output = F.broadcast_mul(
            decoder_output, scale.expand_dims(-1).expand_dims(-1)
        )
        return scaled_decoder_output


class Seq2SeqTrainingNetwork(Seq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        feat_static_cat: Tensor,
        past_feat_dynamic_real: Tensor,
        future_feat_dynamic_real: Tensor,
    ) -> Tensor:
        """

        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: mx.nd.NDArray or mx.sym.Symbol
            past target
        future_target: mx.nd.NDArray or mx.sym.Symbol
            future target
        feat_static_cat: mx.nd.NDArray or mx.sym.Symbol
            static categorical features
        past_feat_dynamic_real: mx.nd.NDArray or mx.sym.Symbol
            past dynamic real-valued features
        future_feat_dynamic_real: mx.nd.NDArray or mx.sym.Symbol
            future dynamic real-valued features

        Returns
        -------
        mx.nd.NDArray or mx.sym.Symbol
           the computed loss
        """
        scaled_decoder_output = self.compute_decoder_outputs(
            F,
            past_target=past_target,
            feat_static_cat=feat_static_cat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            future_feat_dynamic_real=future_feat_dynamic_real,
        )
        projected = self.quantile_proj(scaled_decoder_output)
        loss = self.loss(future_target, projected)
        # TODO: there used to be "nansum" here, to be fully equivalent we
        # TODO: should have a "nanmean" here
        # TODO: shouldn't we sum and divide by the number of observed values
        # TODO: here?
        return loss


class Seq2SeqPredictionNetwork(Seq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        feat_static_cat: Tensor,
        past_feat_dynamic_real: Tensor,
        future_feat_dynamic_real: Tensor,
    ) -> Tensor:
        """

        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: mx.nd.NDArray or mx.sym.Symbol
            past target
        feat_static_cat: mx.nd.NDArray or mx.sym.Symbol
            static categorical features
        past_feat_dynamic_real: mx.nd.NDArray or mx.sym.Symbol
            past dynamic real-valued features
        future_feat_dynamic_real: mx.nd.NDArray or mx.sym.Symbol
            future dynamic real-valued features

        Returns
        -------
        mx.nd.NDArray or mx.sym.Symbol
            the predicted sequence
        """
        scaled_decoder_output = self.compute_decoder_outputs(
            F,
            past_target=past_target,
            feat_static_cat=feat_static_cat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            future_feat_dynamic_real=future_feat_dynamic_real,
        )
        predictions = self.quantile_proj(scaled_decoder_output).swapaxes(2, 1)

        return predictions
