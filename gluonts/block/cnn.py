# Standard library imports
from typing import Optional

# Third-party imports
from mxnet import gluon
from mxnet.gluon import nn


class CausalConv1D(gluon.HybridBlock):
    '''
    1D causal temporal convolution, where the term causal means that output[t]
    does not depend on input[t+1:]. Notice that Conv1D is not implemented in Gluon.

    This is the basic structure used in Wavenet and Temporal Convolution Network. See the following papers for details.

    1. Van Den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner,
    N., Senior, A.W. and Kavukcuoglu, K., 2016, September. WaveNet: A generative model for raw audio. In SSW (p. 125).

    2. Bai, S., Kolter, J.Z. and Koltun, V., 2018.
    An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
    arXiv preprint arXiv:1803.01271.

    The output has the same shape as the input, while we always left pad zeros.

    Parameters
    ----------

    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.

    kernel_size : int or tuple/list of 1 int
        Specifies the dimensions of the convolution window.

    dilation : int or tuple/list of 1 int
        Specifies the dilation rate to use for dilated convolution.

    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).

    '''

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        activation: Optional[str] = 'relu',
        **kwargs,
    ):
        super(CausalConv1D, self).__init__(**kwargs)

        self.dilation = dilation
        self.kernel_size = kernel_size
        self.padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1D(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            activation=activation,
            **kwargs,
        )

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, data):
        """
        In Gluon's conv1D implementation, input has dimension NCW where N is batch_size, C is channel, and W is time.


        Parameters
        ----------
        data : Symbol or NDArray
            Shape (batch_size, num_features, sequence_length)

        Returns
        -------
        Symbol or NDArray
            causal conv1d output. Shape (batch_size, num_features, sequence_length)
        """
        ct = self.conv1d(data)
        if self.kernel_size > 0:
            ct = F.slice_axis(ct, axis=2, begin=0, end=-self.padding)
        return ct


class DilatedCausalGated(gluon.HybridBlock):
    """
    1D convolution with Gated mechanism, see the Wavenet papers described above.

    Parameters
    ----------
    inner_channels : int
        The dimensionality of the intermediate space

    out_channels : int
        The dimensionality of the output space

    kernel_size : int or tuple/list of 1 int
        Specifies the dimensions of the convolution window.

    dilation : int or tuple/list of 1 int
        Specifies the dilation rate to use for dilated convolution.
    """

    def __init__(
        self, inner_channels, out_channels, kernel_size, dilation, **kwargs
    ):
        super(DilatedCausalGated, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = CausalConv1D(
                channels=inner_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                activation="tanh",
            )
            self.conv2 = CausalConv1D(
                channels=inner_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                activation="sigmoid",
            )
            self.output_conv = gluon.nn.Conv1D(
                channels=out_channels, kernel_size=1
            )

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x):
        """
        Parameters
        ----------
        input : Symbol or NDArray
            input features, shape (batch_size, num_features, sequence_length)

        Returns
        -------
        Symbol or NDArray
            output, shape (batch_size, num_features, sequence_length)
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return self.output_conv(x1 * x2)


class ResidualSequential(gluon.nn.HybridSequential):
    """
    Adding residual connection to each layer of the hybrid sequential Blocks
    """

    def __init__(self, **kwargs):
        super(ResidualSequential, self).__init__(**kwargs)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x):
        outs = []
        for i, block in enumerate(self._children.values()):
            out = block(x)
            outs.append(out)
            if i == 0:
                x = out
            else:
                x = x + out

        return sum(outs)
