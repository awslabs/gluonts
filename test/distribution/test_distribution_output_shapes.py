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

from typing import Tuple, List, Union
import pytest
from itertools import product

import mxnet as mx

from gluonts.model.common import Tensor
from gluonts.distribution import (
    DistributionOutput,
    GaussianOutput,
    GammaOutput,
    BetaOutput,
    LaplaceOutput,
    MixtureDistributionOutput,
    MultivariateGaussianOutput,
    LowrankMultivariateGaussianOutput,
    NegativeBinomialOutput,
    PiecewiseLinearOutput,
    PoissonOutput,
    StudentTOutput,
    UniformOutput,
    DirichletOutput,
    DirichletMultinomialOutput,
)


@pytest.mark.parametrize(
    "distr_out, data, loc, scale, expected_batch_shape, expected_event_shape",
    [
        (
            GaussianOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            StudentTOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            GammaOutput(),
            mx.nd.random.gamma(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            BetaOutput(),
            mx.nd.random.gamma(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            MultivariateGaussianOutput(dim=5),
            mx.nd.random.normal(shape=(3, 4, 10)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4),
            (5,),
        ),
        (
            LowrankMultivariateGaussianOutput(dim=5, rank=4),
            mx.nd.random.normal(shape=(3, 4, 10)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4),
            (5,),
        ),
        (
            DirichletOutput(dim=5),
            mx.nd.random.gamma(shape=(3, 4, 5)),
            [None],
            [None],
            (3, 4),
            (5,),
        ),
        (
            DirichletMultinomialOutput(dim=5, n_trials=10),
            mx.nd.random.gamma(shape=(3, 4, 5)),
            [None],
            [None],
            (3, 4),
            (5,),
        ),
        (
            LaplaceOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            NegativeBinomialOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            UniformOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            PiecewiseLinearOutput(num_pieces=3),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            MixtureDistributionOutput([GaussianOutput(), StudentTOutput()]),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            MixtureDistributionOutput(
                [
                    MultivariateGaussianOutput(dim=5),
                    MultivariateGaussianOutput(dim=5),
                ]
            ),
            mx.nd.random.normal(shape=(3, 4, 10)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4),
            (5,),
        ),
        (
            PoissonOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None],
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
    ],
)
def test_distribution_output_shapes(
    distr_out: DistributionOutput,
    data: Tensor,
    loc: List[Union[None, Tensor]],
    scale: List[Union[None, Tensor]],
    expected_batch_shape: Tuple,
    expected_event_shape: Tuple,
):
    args_proj = distr_out.get_args_proj()
    args_proj.initialize()

    args = args_proj(data)

    assert distr_out.event_shape == expected_event_shape

    for l, s in product(loc, scale):

        distr = distr_out.distribution(args, loc=l, scale=s)

        assert distr.batch_shape == expected_batch_shape
        assert distr.event_shape == expected_event_shape

        x = distr.sample()

        assert x.shape == distr.batch_shape + distr.event_shape

        loss = distr.loss(x)

        assert loss.shape == distr.batch_shape

        x1 = distr.sample(num_samples=1)

        assert x1.shape == (1,) + distr.batch_shape + distr.event_shape

        x3 = distr.sample(num_samples=3)

        assert x3.shape == (3,) + distr.batch_shape + distr.event_shape
