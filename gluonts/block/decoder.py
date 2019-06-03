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

# Standard library imports
from typing import List

# Third-party imports
from mxnet.gluon import nn

# First-party imports
from gluonts.block.mlp import MLP
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class Seq2SeqDecoder(nn.HybridBlock):
    """
    Abstract class for the Decoder block in sequence-to-sequence models.
    """

    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, dynamic_input: Tensor, static_input: Tensor
    ) -> None:
        """
        Abstract function definition of the hybrid_forward.

        Parameters
        ----------
        dynamic_input
            dynamic_features, shape (batch_size, sequence_length, num_features)
            or (N, T, C)

        static_input
            static features, shape (batch_size, num_features) or (N, C)

        """
        pass


class ForkingMLPDecoder(Seq2SeqDecoder):
    """
    Multilayer perceptron decoder for sequence-to-sequence models.

    See following paper for details:
        Wen, R., Torkkola, K., and Narayanaswamy, B. (2017).
        A multi-horizon quantile recurrent forecaster.
        arXiv preprint arXiv:1711.11053.

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

    def hybrid_forward(
        self, F, dynamic_input: Tensor, static_input: Tensor = None
    ) -> Tensor:
        """
        ForkingMLPDecoder forward call.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        dynamic_input
            dynamic_features, shape (batch_size, sequence_length, num_features)
            or (N, T, C).

        static_input
            not used in this decoder.

        Returns
        -------
        Tensor
            mlp output, shape (0, 0, dec_len, final_dims).

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
        self,
        F,
        static_input: Tensor,  # (batch_size, static_input_dim)
        dynamic_input: Tensor,  # (batch_size,
    ) -> Tensor:
        """
        OneShotDecoder forward call

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        static_input
            static features, shape (batch_size, num_features) or (N, C)

        dynamic_input
            dynamic_features, shape (batch_size, sequence_length, num_features)
            or (N, T, C)
        Returns
        -------
        Tensor
            mlp output, shape (batch_size, dec_len, size of last layer)
        """
        static_input_tile = self.expander(static_input).reshape(
            (0, self.decoder_length, self.static_outputs_per_time_step)
        )
        combined_input = F.concat(dynamic_input, static_input_tile, dim=2)

        out = self.mlp(combined_input)  # (N, T, layer_sizes[-1])
        return out
