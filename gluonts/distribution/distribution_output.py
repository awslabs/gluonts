# Standard library imports
from typing import Callable, Dict, Optional, Tuple

# Third-party imports
import numpy as np
from mxnet import gluon

# First-party imports
from gluonts.core.component import DType, validated
from gluonts.distribution.bijection import AffineTransformation
from gluonts.model.common import Tensor

# Relative imports
from .distribution import Distribution
from .transformed_distribution import TransformedDistribution


class ArgProj(gluon.HybridBlock):
    r"""
    A block that can be used to project from a dense layer to distribution
    arguments.

    Parameters
    ----------
    dim_args
        Dictionary with string key and int value
        dimension of each arguments that will be passed to the domain
        map, the names are used as parameters prefix.
    domain_map
        Function returning a tuple containing one tensor
        a function or a HybridBlock. This will be called with num_args
        arguments and should return a tuple of outputs that will be
        used when calling the distribution constructor.
    """

    def __init__(
        self,
        args_dim: Dict[str, int],
        domain_map: Callable[..., Tuple[Tensor]],
        float_type: DType = np.float32,
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.args_dim = args_dim
        self.float_type = float_type
        self.proj = [
            gluon.nn.Dense(
                dim,
                flatten=False,
                dtype=self.float_type,
                prefix=f"{prefix}_distr_{name}_",
            )
            for name, dim in args_dim.items()
        ]
        for dense in self.proj:
            self.register_child(dense)
        self.domain_map = domain_map

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, x: Tensor) -> Tuple[Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)


class Output:
    r"""
    Class to connect a network to some output
    """

    args_dim: Dict[str, int]

    def get_args_proj(self, prefix: Optional[str] = None) -> ArgProj:
        return ArgProj(
            args_dim=self.args_dim,
            domain_map=gluon.nn.HybridLambda(self.domain_map),
            prefix=prefix,
        )

    def domain_map(self, F, *args: Tensor):
        raise NotImplementedError()


class DistributionOutput(Output):
    r"""
    Class to construct a distribution given the output of a network.
    """

    distr_cls: type

    @validated()
    def __init__(self) -> None:
        pass

    def distribution(
        self, distr_args, scale: Optional[Tensor] = None
    ) -> Distribution:
        r"""
        Construct the associated distribution, given the collection of
        constructor arguments and, optionally, a scale tensor.
        """
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedDistribution(
                distr, AffineTransformation(scale=scale)
            )

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions
        that this object constructs.
        """
        raise NotImplementedError()

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple,
        of the distributions that this object constructs.
        """
        return len(self.event_shape)

    def domain_map(self, F, *args: Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends
        on the type of distribution, while the correct shape is obtained by
        reshaping the trailing axis in such a way that the returned tensors
        define a distribution of the right event_shape.
        """
        raise NotImplementedError()
