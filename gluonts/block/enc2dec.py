# Standard library imports
from typing import Tuple

# Third-party imports
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


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
        self, F, encoder_output_static: Tensor,
            encoder_output_dynamic: Tensor,
            future_features:  Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------

        encoder_output_static
            shape (batch_size, num_features) or (N, C)

        encoder_output_dynamic
            shape (batch_size, context_length, num_features) or (N, T, C)

        future_features
            shape (batch_size, prediction_length, num_features) or (N, T, C)


        Returns
        -------
        Tensor
            shape (batch_size, num_features) or (N, C)

        Tensor
            shape (batch_size, prediction_length, num_features) or (N, T, C)

        Tensor
            shape (batch_size, sequence_length, num_features) or (N, T, C)

        """
        pass


class PassThroughEnc2Dec(Seq2SeqEnc2Dec):
    """
    Simplest class for passing encoder tensors do decoder. Passes through
    tensors.
    """

    def hybrid_forward(
        self, F, encoder_output_static: Tensor, encoder_output_dynamic: Tensor,
            future_features: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------

        encoder_output_static
            shape (batch_size, num_features) or (N, C)

        encoder_output_dynamic
            shape (batch_size, context_length, num_features) or (N, T, C)

        future_features
            shape (batch_size, prediction_length, num_features) or (N, T, C)


        Returns
        -------
        Tensor
            shape (batch_size, num_features) or (N, C)

        Tensor
            shape (batch_size, prediction_length, num_features) or (N, T, C)

        Tensor
            shape (batch_size, sequence_length, num_features) or (N, T, C)

        """
        return encoder_output_static, encoder_output_dynamic, future_features
