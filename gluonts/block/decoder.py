# Standard library imports
from typing import List

# Third-party imports
from mxnet.gluon import nn

# First-party imports
from gluonts.block.mlp import MLP
from gluonts.core.component import validated


class Seq2SeqDecoder(nn.HybridBlock):
    """
    Abstract class for the Decoder
    """

    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, static_input, dynamic_input) -> None:
        """
        Parameters
        ----------

        static_input : Symbol or NDArray
            static features, shape (batch_size, num_features) or (N, C)
        dynamic_input : Symbol or NDArray
            dynamic_features, shape (batch_size, sequence_length, num_features) or (N, T, C)
        """
        pass


class ForkingMLPDecoder(Seq2SeqDecoder):
    @validated()
    def __init__(
        self,
        dec_len: int,
        final_dim: int,
        hidden_dimension_sequence: List[int] = list([]),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dec_len = dec_len
        self.final_dims = final_dim

        with self.name_scope():
            self.model = nn.HybridSequential()

            for layer_no, layer_dim in enumerate(hidden_dimension_sequence):
                layer = nn.Dense(
                    dec_len * layer_dim,
                    flatten=False,
                    activation='relu',
                    prefix=f"mlp_{layer_no:#02d}'_",
                )
                self.model.add(layer)

            layer = nn.Dense(
                dec_len * final_dim,
                flatten=False,
                activation='relu',
                prefix=f"mlp_{len(hidden_dimension_sequence):#02d}'_",
            )
            self.model.add(layer)

    def hybrid_forward(self, F, dynamic_input, static_input=None):
        mlp_output = self.model(dynamic_input)
        mlp_output = mlp_output.reshape(
            shape=(0, 0, self.dec_len, self.final_dims)
        )
        return mlp_output


class OneShotDecoder(Seq2SeqDecoder):
    @validated()
    def __init__(
        self,
        decoder_length: int,
        layer_sizes: List[int],
        static_outputs_per_time_step: int,
    ) -> None:
        super().__init__()
        self.decoder_length = decoder_length
        self.static_outputs_per_time_step = static_outputs_per_time_step
        with self.name_scope():
            self.mlp = MLP(layer_sizes, flatten=False)
            self.expander = nn.Dense(
                units=decoder_length * static_outputs_per_time_step
            )

    def hybrid_forward(
        self,
        F,
        static_input,  # (batch_size, static_input_dim)
        dynamic_input,  # (batch_size, decoder_length, dynamic_input_dim)
    ):
        static_input_tile = self.expander(static_input).reshape(
            (0, self.decoder_length, self.static_outputs_per_time_step)
        )
        combined_input = F.concat(dynamic_input, static_input_tile, dim=2)

        out = self.mlp(combined_input)  # (N, T, layer_sizes[-1])
        return out
