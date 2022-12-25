from torch_extensions.softmax_rnn import SoftmaxRNN


class SwitchTransitionModelDeterministicRNN(SoftmaxRNN):
    def __init__(self, config):
        super().__init__(
            dim_in=config.dims.auxiliary,
            dim_hidden=config.n_hidden_rnn,
            dim_out=config.dims.switch,
        )
