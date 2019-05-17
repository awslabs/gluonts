# Standard library imports
from typing import Tuple

# Third-party imports
import mxnet as mx
from mxnet import gluon

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor

# Relative imports
from .distribution import Distribution, _sample_multiple, getF
from .distribution_output import DistributionOutput


class Binned(Distribution):
    is_reparameterizable = False

    def __init__(
        self, bin_probs: Tensor, bin_edges: Tensor, bin_values: Tensor, F=None
    ) -> None:
        """
        A binned distribution that represents a set of bins via
        bin edges e_i and values v_i

        bins:    [e_0, e1), [e_1, e_2), , ..., [e_{n-1}, e_n)
        values:      v0,        v1,       ...,      v_{n-1}
        prob:        p0,        p1,       ...,      p_{n-1}

        :param bin_edges: 1d array of bin edges
        :param bin_values: 1d array of values representing the bins.
          This should have one more entry than bin_edges
        :param bin_probs: 2d array of probability per bin
        """
        self.bin_edges = bin_edges
        self.bin_values = bin_values
        self.bin_probs = bin_probs
        self.F = F if F else getF(bin_probs)

    @property
    def batch_shape(self) -> Tuple:
        return self.bin_values.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def mean(self):
        return (self.bin_probs * self.bin_values).sum(axis=-1)

    @property
    def stddev(self):
        Ex2 = (self.bin_probs * self.bin_values.square()).sum(axis=-1)
        return (Ex2 - self.mean.square()).sqrt()

    def log_prob(self, x):
        # reshape first to capture both (d,) and (d,1) cases and then add extra dimension
        x = x.reshape(0, -1).expand_dims(axis=-1)
        # TODO: when mxnet has searchsorted replace this
        left_edges = self.bin_edges.slice_axis(axis=-1, begin=0, end=-1)
        right_edges = self.bin_edges.slice_axis(axis=-1, begin=1, end=None)
        mask = self.F.broadcast_lesser_equal(
            left_edges.expand_dims(axis=-2), x
        ) * self.F.broadcast_lesser(x, right_edges.expand_dims(axis=-2))
        return (self.bin_probs.log() * mask).sum(axis=-1)

    def sample(self, num_samples=None):
        def s(bin_probs):
            F = self.F
            indices = F.sample_multinomial(bin_probs)
            if num_samples is None:
                return self.bin_values.pick(indices, -1).reshape_like(
                    F.zeros_like(indices.astype('float32'))
                )
            else:
                return F.repeat(
                    F.expand_dims(self.bin_values, axis=0),
                    repeats=num_samples,
                    axis=0,
                ).pick(indices, -1)

        return _sample_multiple(s, self.bin_probs, num_samples=num_samples)


class BinnedArgs(gluon.HybridBlock):
    def __init__(
        self, bin_edges: mx.nd.NDArray, bin_values: mx.nd.NDArray, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        with self.name_scope():
            self.bin_edges = self.params.get_constant('binedges', bin_edges)
            self.bin_values = self.params.get_constant('binvalues', bin_values)
            self.num_bins = bin_values.shape[0]

            # needs to be named self.proj for consistency with the ArgProj class and the inference tests
            self.proj = gluon.nn.HybridSequential()
            self.proj.add(
                gluon.nn.Dense(
                    self.num_bins,
                    prefix='binproj',
                    flatten=False,
                    weight_initializer=mx.init.Xavier(),
                )
            )
            self.proj.add(gluon.nn.HybridLambda('softmax'))

    def hybrid_forward(
        self, F, x: Tensor, bin_values: Tensor, bin_edges: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ps = self.proj(x)
        return (
            ps.reshape((-2, -1, self.num_bins), reverse=1),
            # For some reason hybridize does not work when returning constants directly
            bin_edges + 0.0,
            bin_values + 0.0,
        )


class BinnedOutput(DistributionOutput):
    distr_cls: type = Binned

    @validated()
    def __init__(self, bin_values: list, bin_edges: list = None) -> None:
        # cannot pass directly nd.array because it is not serializable
        bv = mx.nd.array(bin_values)
        assert len(bv.shape) == 1
        self.bin_values = bv

        if bin_edges is not None:
            be = mx.nd.array(bin_edges)
            assert len(be.shape) == 1
            assert be.shape[0] == self.bin_values.shape[0] + 1
            self.bin_edges = be
        else:
            means = (bv[1:] + bv[:-1]) / 2.0
            self.bin_edges = mx.nd.concatenate(
                [
                    mx.nd.array([-1.0E10]),
                    mx.nd.array(means),
                    mx.nd.array([1E10]),
                ]
            )

        self.num_bins = self.bin_values.shape[0]
        for i in range(self.num_bins):
            assert (
                self.bin_edges[i] < self.bin_values[i] < self.bin_edges[i + 1]
            )

    def get_args_proj(self, *args, **kwargs) -> gluon.nn.HybridBlock:
        return BinnedArgs(bin_edges=self.bin_edges, bin_values=self.bin_values)

    def distribution(self, args, scale=None) -> Binned:
        probs, edges, values = args
        if scale is not None:
            F = getF(probs)
            edges = F.broadcast_mul(edges, scale)
            values = F.broadcast_mul(values, scale)
        return Binned(probs, edges, values)

    @property
    def event_shape(self) -> Tuple:
        return ()
