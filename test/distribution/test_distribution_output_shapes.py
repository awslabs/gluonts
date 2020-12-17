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

# type: ignore


import pytest
from itertools import product
from typing import List, Tuple, Union

import mxnet as mx

from gluonts.mx import Tensor
from gluonts.mx.distribution import (
    BetaOutput,
    DeterministicOutput,
    DirichletMultinomialOutput,
    DirichletOutput,
    DistributionOutput,
    GammaOutput,
    GaussianOutput,
    LaplaceOutput,
    LowrankMultivariateGaussianOutput,
    MixtureDistributionOutput,
    MultivariateGaussianOutput,
    NegativeBinomialOutput,
    PiecewiseLinearOutput,
    PoissonOutput,
    StudentTOutput,
    UniformOutput,
)

TEST_CASES = [
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
        MixtureDistributionOutput([GaussianOutput(), StudentTOutput()]),
        mx.nd.random.normal(shape=(3, 4, 5, 6)),
        [None, mx.nd.ones(shape=(3, 4, 5))],
        [None, mx.nd.ones(shape=(3, 4, 5))],
        (3, 4, 5),
        (),
    ),
    (
        PoissonOutput(),
        mx.nd.random.normal(shape=(3, 4, 5, 6)),
        [None],
        [None, mx.nd.ones(shape=(3, 4, 5))],
        (3, 4, 5),
        (),
    ),
    (
        DeterministicOutput(42.0),
        mx.nd.random.normal(shape=(3, 4, 5, 6)),
        [None, mx.nd.ones(shape=(3, 4, 5))],
        [None, mx.nd.ones(shape=(3, 4, 5))],
        (3, 4, 5),
        (),
    ),
    (
        MultivariateGaussianOutput(dim=5),
        mx.nd.random.normal(shape=(3, 4, 10)),
        [None, mx.nd.ones(shape=(3, 4, 5))],
        [None],
        (3, 4),
        (5,),
    ),
    (
        LowrankMultivariateGaussianOutput(dim=5, rank=4),
        mx.nd.random.normal(shape=(3, 4, 10)),
        [None, mx.nd.ones(shape=(3, 4, 5))],
        [None],
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
]

TEST_CASES_WITHOUT_VARIANCE = [
    (
        MixtureDistributionOutput(
            [
                MultivariateGaussianOutput(dim=5),
                MultivariateGaussianOutput(dim=5),
            ]
        ),
        mx.nd.random.normal(shape=(3, 4, 10)),
        [None, mx.nd.ones(shape=(3, 4, 5))],
        [None],
        (3, 4),
        (5,),
    ),
]

TEST_CASES_WITHOUT_MEAN_NOR_VARIANCE = [
    (
        PiecewiseLinearOutput(num_pieces=3),
        mx.nd.random.normal(shape=(3, 4, 5, 6)),
        [None, mx.nd.ones(shape=(3, 4, 5))],
        [None, mx.nd.ones(shape=(3, 4, 5))],
        (3, 4, 5),
        (),
    ),
]


@pytest.mark.parametrize(
    "distr_out, data, loc, scale, expected_batch_shape, expected_event_shape",
    TEST_CASES
    + TEST_CASES_WITHOUT_VARIANCE
    + TEST_CASES_WITHOUT_MEAN_NOR_VARIANCE,
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


@pytest.mark.parametrize(
    "distr_out, data, loc, scale, expected_batch_shape, expected_event_shape",
    TEST_CASES + TEST_CASES_WITHOUT_VARIANCE,
)
def test_distribution_output_mean(
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

    for l, s in product(loc, scale):
        distr = distr_out.distribution(args, loc=l, scale=s)
        assert distr.mean.shape == expected_batch_shape + expected_event_shape


@pytest.mark.parametrize(
    "distr_out, data, loc, scale, expected_batch_shape, expected_event_shape",
    TEST_CASES,
)
def test_distribution_output_variance(
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

    for l, s in product(loc, scale):
        distr = distr_out.distribution(args, loc=l, scale=s)
        assert (
            distr.variance.shape
            == expected_batch_shape + expected_event_shape * 2
        )


@pytest.mark.parametrize("value, model_output_shape", [(42.0, (3, 4, 5))])
def test_deterministic_output(value: float, model_output_shape):
    do = DeterministicOutput(value)
    x = mx.nd.ones(model_output_shape)

    args_proj = do.get_args_proj()
    args_proj.initialize()
    args = args_proj(x)
    distr = do.distribution(args)

    s = distr.sample()

    assert (
        (s == value * mx.nd.ones(shape=model_output_shape[:-1]))
        .asnumpy()
        .all()
    )

    assert (distr.prob(s) == 1.0).asnumpy().all()

    s10 = distr.sample(10)

    assert (
        (s10 == value * mx.nd.ones(shape=(10,) + model_output_shape[:-1]))
        .asnumpy()
        .all()
    )

    assert (distr.prob(s10) == 1.0).asnumpy().all()
