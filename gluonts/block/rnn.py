# Third-party imports
from mxnet.gluon import HybridBlock, rnn

# First-party imports
from gluonts.core.component import validated


class RNN(HybridBlock):
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
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(
                    num_hidden,
                    num_layers,
                    bidirectional=bidirectional,
                    activation='relu',
                    layout='NTC',
                )
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(
                    num_hidden,
                    num_layers,
                    bidirectional=bidirectional,
                    layout='NTC',
                )
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(
                    num_hidden,
                    num_layers,
                    bidirectional=bidirectional,
                    layout='NTC',
                )
            elif mode == 'gru':
                self.rnn = rnn.GRU(
                    num_hidden,
                    num_layers,
                    bidirectional=bidirectional,
                    layout='NTC',
                )
            else:
                raise ValueError(
                    "Invalid mode %s. Options are rnn_relu, rnn_tanh, lstm, and gru "
                    % mode
                )

    def hybrid_forward(self, F, inputs):  # NTC in, NTC out
        return self.rnn(inputs)
