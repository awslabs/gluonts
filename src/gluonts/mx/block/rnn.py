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
from mxnet.gluon import HybridBlock, rnn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class RNN(HybridBlock):
    """
    Defines an RNN block.

    Parameters
    ----------
    mode
        type of the RNN. Can be either: rnn_relu (RNN with relu activation),
        rnn_tanh, (RNN with tanh activation), lstm or gru.

    num_hidden
        number of units per hidden layer.

    num_layers
        number of hidden layers.

    bidirectional
        toggle use of bi-directional RNN as encoder.
    """

    @validated()
    def __init__(
        self,
        mode: str,
        num_hidden: int,
        num_layers: int,
        bidirectional: bool = False,
        **kwargs,
    ):
        super(RNN, self).__init__(**kwargs)

        with self.name_scope():
            if mode == "rnn_relu":
                self.rnn = rnn.RNN(
                    num_hidden,
                    num_layers,
                    bidirectional=bidirectional,
                    activation="relu",
                    layout="NTC",
                )
            elif mode == "rnn_tanh":
                self.rnn = rnn.RNN(
                    num_hidden,
                    num_layers,
                    bidirectional=bidirectional,
                    layout="NTC",
                )
            elif mode == "lstm":
                self.rnn = rnn.LSTM(
                    num_hidden,
                    num_layers,
                    bidirectional=bidirectional,
                    layout="NTC",
                )
            elif mode == "gru":
                self.rnn = rnn.GRU(
                    num_hidden,
                    num_layers,
                    bidirectional=bidirectional,
                    layout="NTC",
                )
            else:
                raise ValueError(
                    "Invalid mode %s. Options are rnn_relu, rnn_tanh, lstm, and gru "
                    % mode
                )

    def hybrid_forward(self, F, inputs: Tensor) -> Tensor:  # NTC in, NTC out
        """

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        inputs
            input tensor with shape (batch_size, num_timesteps, num_dimensions)

        Returns
        -------
        Tensor
            rnn output with shape (batch_size, num_timesteps, num_dimensions)
        """
        return self.rnn(inputs)
