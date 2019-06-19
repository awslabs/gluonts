# Standard library imports
from typing import Optional

# First-party imports
from gluonts.block.decoder import Seq2SeqDecoder
from gluonts.block.enc2dec import PassThroughEnc2Dec
from gluonts.block.encoder import Seq2SeqEncoder
from gluonts.block.quantile_output import QuantileOutput
from gluonts.core.component import validated
from gluonts.model.estimator import GluonEstimator
from gluonts.model.forecast import QuantileForecast, Quantile
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.support.util import copy_parameters
from gluonts.trainer import Trainer
from gluonts.transform import (
    AsNumpyArray,
    Chain,
    FieldName,
    TestSplitSampler,
    Transformation,
)

# Relative imports
from ._forking_network import (
    ForkingSeq2SeqPredictionNetwork,
    ForkingSeq2SeqTrainingNetwork,
)
from ._transform import ForkingSequenceSplitter


class ForkingSeq2SeqEstimator(GluonEstimator):
    r"""
    Sequence-to-Sequence (seq2seq) structure with the so-called
    "Forking Sequence" proposed in [Wen2017]_.

    The basic idea is that, given a sequence :math:`x_1, x_2, \cdots, x_T`,
    with a decoding length :math:`\tau`, we learn a NN that solves the
    following series of seq2seq problems:

    .. math::
       :nowrap:

       \begin{eqnarray}
       x_1                     & \mapsto & x_{2:2+\tau}\\
       x_1, x_2                & \mapsto & x_{3:3+\tau}\\
       x_1, x_2, x_3           & \mapsto & x_{4:4+\tau}\\
                               & \ldots  & \\
       x_1, \ldots, x_{T-\tau} & \mapsto & x_{T-\tau+1:T}
       \end{eqnarray}

    Essentially, this means instead of having one cut in the standard seq2seq,
    one has multiple cuts that progress linearly.

    .. [Wen2017] Wen, Ruofeng, et al. "A multi-horizon quantile recurrent
                 forecaster." arXiv preprint arXiv:1711.11053 (2017).

    Parameters
    ----------
    encoder
        seq2seq encoder
    decoder
        seq2seq decoder
    quantile_output
        quantile output
    freq
        frequency of the time series
    prediction_length
        length of the decoding sequence
    context_length
        length of the encoding sequence (prediction_length is used if None)
    trainer
        trainer
    """

    @validated()
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        decoder: Seq2SeqDecoder,
        quantile_output: QuantileOutput,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
    ) -> None:
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"

        super().__init__(trainer=trainer)

        self.encoder = encoder
        self.decoder = decoder
        self.quantile_output = quantile_output
        self.prediction_length = prediction_length
        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            trans=[
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                ForkingSequenceSplitter(
                    train_sampler=TestSplitSampler(),
                    enc_len=self.context_length,
                    dec_len=self.prediction_length,
                ),
            ]
        )

    def create_training_network(self) -> ForkingSeq2SeqTrainingNetwork:
        return ForkingSeq2SeqTrainingNetwork(
            encoder=self.encoder,
            enc2dec=PassThroughEnc2Dec(),
            decoder=self.decoder,
            quantile_output=self.quantile_output,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: ForkingSeq2SeqTrainingNetwork,
    ) -> Predictor:
        # todo: this is specific to quantile output
        quantile_strs = [
            Quantile.from_float(quantile).name
            for quantile in self.quantile_output.quantiles
        ]

        prediction_network = ForkingSeq2SeqPredictionNetwork(
            encoder=trained_network.encoder,
            enc2dec=trained_network.enc2dec,
            decoder=trained_network.decoder,
            quantile_output=trained_network.quantile_output,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            forecast_cls_name=QuantileForecast.__name__,
            forecast_kwargs=dict(forecast_keys=quantile_strs),
        )
