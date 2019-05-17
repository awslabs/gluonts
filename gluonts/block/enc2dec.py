# Third-party imports
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated


class Seq2SeqEnc2Dec(nn.HybridBlock):
    """
    Abstract class for any module that pass encoder to decoder, such as
    attention network.
    """

    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, encoder_output_static, encoder_output_dynamic, future_features
    ):
        """
        Parameters
        ----------

        encoder_output_static : Symbol or NDArray
            shape (batch_size, num_features) or (N, C)

        encoder_output_dynamic : Symbol or NDArray
            shape (batch_size, context_length, num_features) or (N, T, C)

        future_features : Symbol or NDArray
            shape (batch_size, prediction_length, num_features) or (N, T, C)


        Returns
        -------
        decoder_input_static : Symbol or NDArray
            shape (batch_size, num_features) or (N, C)

        decoder_inuput_dynamic : Symbol or NDArray
            shape (batch_size, prediction_length, num_features) or (N, T, C)

        future_features : Symbol or NDArray
            shape (batch_size, sequence_length, num_features) or (N, T, C)

        """
        pass


class PassThroughEnc2Dec(Seq2SeqEnc2Dec):
    """
    passing through, Noop
    """

    def hybrid_forward(
        self, F, encoder_output_static, encoder_output_dynamic, future_features
    ):
        return encoder_output_static, encoder_output_dynamic, future_features
