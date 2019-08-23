from mxnet.gluon import HybridBlock, nn

from gluonts.block.rnn import RNN


class RNNModel(HybridBlock):
    def __init__(
        self,
        mode,
        num_hidden,
        num_layers,
        num_output,
        bidirectional=False,
        **kwargs,
    ):
        super(RNNModel, self).__init__(**kwargs)
        self.num_output = num_output

        with self.name_scope():
            self.rnn = RNN(
                mode=mode,
                num_hidden=num_hidden,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )

            self.decoder = nn.Dense(
                num_output, in_units=num_hidden, flatten=False
            )

    def hybrid_forward(self, F, inputs):
        return self.decoder(self.rnn(inputs))
