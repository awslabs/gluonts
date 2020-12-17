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

from collections import ChainMap
from typing import List, Optional, Tuple

import numpy as np
from mxnet import gluon

from gluonts.core.component import validated
from gluonts.mx import Tensor

from . import Distribution
from .bijection import AffineTransformation
from .bijection_output import BijectionOutput
from .distribution_output import ArgProj, DistributionOutput
from .transformed_distribution import TransformedDistribution


class TransformedDistributionOutput(DistributionOutput):
    r"""
    Class to connect a network to a distribution that is transformed
    by a sequence of learnable bijections.
    """

    @validated()
    def __init__(
        self,
        base_distr_output: DistributionOutput,
        transforms_output: List[BijectionOutput],
    ) -> None:
        super().__init__()
        self.base_distr_output = base_distr_output
        self.transforms_output = transforms_output

        self.base_distr_args_dim = base_distr_output.args_dim
        self.transforms_args_dim = [
            transform.args_dim for transform in transforms_output
        ]

        def _fuse(t1: Tuple, t2: Tuple) -> Tuple:
            if len(t1) > len(t2):
                t1, t2 = t2, t1
            # from here on len(t2) >= len(t1)
            assert t2[-len(t1) :] == t1
            return t2

        self._event_shape: Tuple[int, ...] = ()
        for to in self.transforms_output:
            self._event_shape = _fuse(self._event_shape, to.event_shape)

    def get_args_proj(self, prefix: Optional[str] = None) -> ArgProj:
        return ArgProj(
            args_dim=dict(
                self.base_distr_args_dim,
                **dict(ChainMap(*self.transforms_args_dim)),
            ),
            domain_map=gluon.nn.HybridLambda(self.domain_map),
            prefix=prefix,
        )

    def _split_args(self, args):
        # Since hybrid_forward does not support dictionary,
        # we have to separate the raw outputs of the network based on the indices
        # and map them to the learnable parameters
        num_distr_args = len(self.base_distr_args_dim)
        distr_args = args[0:num_distr_args]

        num_transforms_args = [
            len(transform_dim_args)
            for transform_dim_args in self.transforms_args_dim
        ]
        # starting indices of arguments for each transformation
        num_args_cumsum = np.cumsum([num_distr_args] + num_transforms_args)

        # get the arguments for each of the transformations
        transforms_args = list(
            map(
                lambda ixs: args[ixs[0] : ixs[1]],
                zip(num_args_cumsum, num_args_cumsum[1:]),
            )
        )

        return distr_args, transforms_args

    def domain_map(self, F, *args: Tensor):
        distr_args, transforms_args = self._split_args(args)

        distr_params = self.base_distr_output.domain_map(F, *distr_args)
        transforms_params = [
            transform_output.domain_map(F, *transform_args)
            for transform_output, transform_args in zip(
                self.transforms_output, transforms_args
            )
        ]

        # flatten the nested tuple
        return sum(tuple([distr_params] + transforms_params), ())

    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> Distribution:
        distr_args, transforms_args = self._split_args(distr_args)
        distr = self.base_distr_output.distr_cls(*distr_args)
        transforms = [
            transform_output.bij_cls(*bij_args)
            for transform_output, bij_args in zip(
                self.transforms_output, transforms_args
            )
        ]

        trans_distr = TransformedDistribution(distr, transforms)

        # Apply scaling as well at the end if scale is not None!
        if loc is None and scale is None:
            return trans_distr
        else:
            return TransformedDistribution(
                trans_distr, [AffineTransformation(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return self._event_shape
