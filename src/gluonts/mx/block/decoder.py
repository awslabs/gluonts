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

from typing import List, Optional

from mxnet.gluon import nn

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.mlp import MLP


class Seq2SeqDecoder(nn.HybridBlock):
    """
    Abstract class for the Decoder block in sequence-to-sequence models.
    """

    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, static_input: Tensor, dynamic_input: Tensor
    ) -> Tensor:
        """
        Abstract function definition of the hybrid_forward.

        Parameters
        ----------
        static_input
            static features, shape (batch_size, channels_seq[-1] + 1) or (N, C)

        dynamic_input
            dynamic_features, shape (batch_size, sequence_length, channels_seq[-1]
            + 1 + decoder_length * num_feat_dynamic)
            or (N, T, C)
        """
        raise NotImplementedError


# TODO: add support for static variables at some point
class ForkingMLPDecoder(Seq2SeqDecoder):
    """
    Multilayer perceptron decoder for sequence-to-sequence models.

    See [WTN+17]_ for details.

    Parameters
    ----------
    dec_len
        length of the decoder (usually the number of forecasted time steps).

    final_dim
        dimensionality of the output per time step (number of predicted
        quantiles).

    hidden_dimension_sequence
        number of hidden units for each MLP layer.
    """

    @validated()
    def __init__(
        self,
        dec_len: int,
        final_dim: int,
        hidden_dimension_sequence: List[int] = [],
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
                    activation="relu",
                    prefix=f"mlp_{layer_no:#02d}'_",
                )
                self.model.add(layer)

            layer = nn.Dense(
                dec_len * final_dim,
                flatten=False,
                activation="softrelu",
                prefix=f"mlp_{len(hidden_dimension_sequence):#02d}'_",
            )
            self.model.add(layer)

    # TODO: add support for static input at some point
    def hybrid_forward(
        self, F, static_input: Tensor, dynamic_input: Tensor
    ) -> Tensor:
        """
        ForkingMLPDecoder forward call.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        static_input
            not used in this decoder.
        dynamic_input
            dynamic_features, shape (batch_size, sequence_length, num_features) or (N, T, C)
            where sequence_length is equal to the encoder length, and num_features is equal
            to channels_seq[-1] + 1 + decoder_length * num_feat_dynamic for the MQ-CNN for example.

        Returns
        -------
        Tensor
            mlp output, shape (batch_size, sequence_length, decoder_length, decoder_mlp_dim_seq[0]).

        """
        mlp_output = self.model(dynamic_input)
        mlp_output = mlp_output.reshape(
            shape=(0, 0, self.dec_len, self.final_dims)
        )
        return mlp_output


class OneShotDecoder(Seq2SeqDecoder):
    """
    OneShotDecoder.

    Parameters
    ----------
    decoder_length
        length of the decoder (number of time steps)
    layer_sizes
        dimensions of the hidden layers
    static_outputs_per_time_step
        number of outputs per time step
    """

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
        self, F, static_input: Tensor, dynamic_input: Tensor
    ) -> Tensor:
        """
        OneShotDecoder forward call

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        static_input
            static features, shape (batch_size, channels_seq[-1] + 1) or (N, C)

        dynamic_input
            dynamic_features, shape (batch_size, sequence_length, channels_seq[-1]
            + 1 + decoder_length * num_feat_dynamic)
            or (N, T, C)

        Returns
        -------
        Tensor
            mlp output, shape (batch_size, decoder_length, size of last layer)
        """
        static_input_tile = self.expander(static_input).reshape(
            (0, self.decoder_length, self.static_outputs_per_time_step)
        )
        combined_input = F.concat(dynamic_input, static_input_tile, dim=2)

        out = self.mlp(combined_input)  # (N, T, layer_sizes[-1])
        return out
