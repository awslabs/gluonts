# Standard library imports
from typing import List, Tuple

# Third-party imports
from mxnet.gluon import nn

# First-party imports
from gluonts.block.cnn import CausalConv1D
from gluonts.block.mlp import MLP
from gluonts.block.rnn import RNN
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class Seq2SeqEncoder(nn.HybridBlock):
    """
    Abstract class for the encoder. An encoder takes a `target` sequence with
    corresponding covariates and maps it into a static latent and
    a dynamic latent code with the same length as the `target` sequence.
    """

    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        F: mx.symbol or mx.ndarray
            Gluon function space

        target : Symbol or NDArray
            target time series,
            shape (batch_size, sequence_length)

        static_features : Symbol or NDArray
            static features,
            shape (batch_size, num_static_features)

        dynamic_features : Symbol or NDArray
            dynamic_features,
            shape (batch_size, sequence_length, num_dynamic_features)


        Returns
        -------
        static_code : Symbol or NDArray
            static code,
            shape (batch_size, num_static_features)

        dynamic_code : Symbol or NDArray
            dynamic code,
            shape (batch_size, sequence_length, num_dynamic_features)
        """
        raise NotImplementedError

    @staticmethod
    def _assemble_inputs(
        F, target: Tensor, static_features: Tensor, dynamic_features: Tensor
    ) -> Tensor:
        """
        Assemble features from target, static features, and the dynamic
        features.

        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space

        target : Symbol or NDArray
            target time series,
            shape (batch_size, sequence_length)

        static_features : Symbol or NDArray
            static features,
            shape (batch_size, num_static_features)

        dynamic_features : Symbol or NDArray
            dynamic_features,
            shape (batch_size, sequence_length, num_dynamic_features)

        Returns
        -------
        inputs : Symbol or NDArray
            combined features,
            shape (batch_size, sequence_length,
                   num_static_features + num_dynamic_features + 1)

        """
        target = target.expand_dims(axis=-1)  # (N, T, 1)

        helper_ones = F.ones_like(target)  # Ones of (N, T, 1)
        tiled_static_features = F.batch_dot(
            helper_ones, static_features.expand_dims(1)
        )  # (N, T, C)
        inputs = F.concat(
            target, tiled_static_features, dynamic_features, dim=2
        )  # (N, T, C)
        return inputs


class HierarchicalCausalConv1DEncoder(Seq2SeqEncoder):
    @validated()
    def __init__(
        self,
        dilation_seq: List[int],
        kernel_size_seq: List[int],
        channels_seq: List[int],
        use_residual: bool = False,
        is_expand_dim: bool = False,  # TODO: name is a bit misleading
        **kwargs,
    ) -> None:
        assert all(
            [x > 0 for x in dilation_seq]
        ), "`dilation_seq` values must be greater than zero"
        assert all(
            [x > 0 for x in kernel_size_seq]
        ), "`kernel_size_seq` values must be greater than zero"
        assert all(
            [x > 0 for x in channels_seq]
        ), "`channel_dim_seq` values must be greater than zero"

        super().__init__(**kwargs)

        self.use_residual = use_residual
        self.is_expand_dim = is_expand_dim
        self.cnn = nn.HybridSequential()

        it = zip(channels_seq, kernel_size_seq, dilation_seq)
        for layer_no, (channels, kernel_size, dilation) in enumerate(it):
            convolution = CausalConv1D(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                activation='relu',
                prefix=f"conv_{layer_no:#02d}'_",
            )
            self.cnn.add(convolution)

    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        if self.is_expand_dim:
            inputs = Seq2SeqEncoder._assemble_inputs(
                F,
                target=target,
                static_features=static_features,
                dynamic_features=dynamic_features,
            )
        else:
            inputs = target

        # NTC -> NCT (or NCW)
        ct = inputs.swapaxes(1, 2)
        ct = self.cnn(ct)
        ct = ct.swapaxes(1, 2)

        # now we are back in NTC
        if self.use_residual:
            ct = F.concat(ct, target, dim=2)

        # return the last state as the static code
        static_code = F.slice_axis(ct, axis=1, begin=-1, end=None)
        static_code = F.squeeze(static_code, axis=1)
        return static_code, ct


class RNNEncoder(Seq2SeqEncoder):
    @validated()
    def __init__(
        self,
        mode: str,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        **kwargs,
    ) -> None:
        assert num_layers > 0, "`num_layers` value must be greater than zero"
        assert hidden_size > 0, "`hidden_size` value must be greater than zero"

        super().__init__(**kwargs)

        with self.name_scope():
            self.rnn = RNN(mode, hidden_size, num_layers, bidirectional)

    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        dynamic_code = self.rnn(target)
        static_code = F.slice_axis(dynamic_code, axis=1, begin=-1, end=None)
        return static_code, dynamic_code


class MLPEncoder(Seq2SeqEncoder):
    @validated()
    def __init__(self, layer_sizes: List[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = MLP(layer_sizes, flatten=True)

    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        inputs = Seq2SeqEncoder._assemble_inputs(
            F, target, static_features, dynamic_features
        )
        static_code = self.model(inputs)
        dynamic_code = F.zeros_like(target).expand_dims(2)
        return static_code, dynamic_code


class LSTMEncoder(Seq2SeqEncoder):
    @validated()
    def __init__(
        self,
        mode: str,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        **kwargs,
    ) -> None:
        assert num_layers > 0, "`num_layers` value must be greater than zero"
        assert hidden_size > 0, "`hidden_size` value must be greater than zero"

        super().__init__(**kwargs)

        with self.name_scope():
            self.rnn = RNN(mode, hidden_size, num_layers, bidirectional)

    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        inputs = Seq2SeqEncoder._assemble_inputs(
            F, target, static_features, dynamic_features
        )
        dynamic_code = self.rnn(inputs)

        # using the last state as the static code,
        # but not working as well as the concat of all the previous states
        static_code = F.squeeze(
            F.slice_axis(dynamic_code, axis=1, begin=-1, end=None), axis=1
        )

        return static_code, dynamic_code
