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

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from mxnet import gluon

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor

from .bijection import AffineTransformation
from .distribution import Distribution
from .transformed_distribution import AffineTransformedDistribution


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
        dtype: DType = np.float32,
        prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.dtype = dtype
        self.proj = [
            gluon.nn.Dense(
                dim,
                flatten=False,
                dtype=self.dtype,
                prefix=f"{prefix}_distr_{name}_",
            )
            for name, dim in args_dim.items()
        ]
        for dense in self.proj:
            self.register_child(dense)
        self.domain_map = domain_map

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, x: Tensor, **kwargs) -> Tuple[Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)


class Output:
    r"""
    Class to connect a network to some output
    """

    args_dim: Dict[str, int]
    _dtype: DType = np.float32

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DType):
        self._dtype = dtype

    def get_args_proj(self, prefix: Optional[str] = None) -> gluon.HybridBlock:
        return ArgProj(
            args_dim=self.args_dim,
            domain_map=gluon.nn.HybridLambda(self.domain_map),
            prefix=prefix,
            dtype=self.dtype,
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
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> Distribution:
        r"""
        Construct the associated distribution, given the collection of
        constructor arguments and, optionally, a scale tensor.

        Parameters
        ----------
        distr_args
            Constructor arguments for the underlying Distribution type.
        loc
            Optional tensor, of the same shape as the
            batch_shape+event_shape of the resulting distribution.
        scale
            Optional tensor, of the same shape as the
            batch_shape+event_shape of the resulting distribution.
        """
        if loc is None and scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return AffineTransformedDistribution(distr, loc=loc, scale=scale)

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

    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the
        log-loss of the corresponding distribution. By default 0.0.
        This value will be used when padding data series.
        """
        return 0.0

    def domain_map(self, F, *args: Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends
        on the type of distribution, while the correct shape is obtained by
        reshaping the trailing axis in such a way that the returned tensors
        define a distribution of the right event_shape.
        """
        raise NotImplementedError()
