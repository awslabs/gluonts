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

from typing import Tuple

import mxnet as mx
import pytest

from gluonts.model.tpp.distribution import Loglogistic, Weibull
from gluonts.mx.distribution import (
    Beta,
    Dirichlet,
    DirichletMultinomial,
    Distribution,
    Gamma,
    Gaussian,
    Laplace,
    MixtureDistribution,
    MultivariateGaussian,
    NegativeBinomial,
    PiecewiseLinear,
    Poisson,
    StudentT,
    TransformedDistribution,
    Uniform,
)
from gluonts.mx.distribution.bijection import AffineTransformation
from gluonts.mx.distribution.box_cox_transform import BoxCoxTransform
from gluonts.mx.util import make_nd_diag


@pytest.mark.parametrize(
    "distr, expected_batch_shape, expected_event_shape",
    [
        (
            Gaussian(
                mu=mx.nd.zeros(shape=(3, 4, 5)),
                sigma=mx.nd.ones(shape=(3, 4, 5)),
            ),
            (3, 4, 5),
            (),
        ),
        (
            Gamma(
                alpha=mx.nd.ones(shape=(3, 4, 5)),
                beta=mx.nd.ones(shape=(3, 4, 5)),
            ),
            (3, 4, 5),
            (),
        ),
        (
            Beta(
                alpha=mx.nd.ones(shape=(3, 4, 5)),
                beta=mx.nd.ones(shape=(3, 4, 5)),
            ),
            (3, 4, 5),
            (),
        ),
        (
            StudentT(
                mu=mx.nd.zeros(shape=(3, 4, 5)),
                sigma=mx.nd.ones(shape=(3, 4, 5)),
                nu=mx.nd.ones(shape=(3, 4, 5)),
            ),
            (3, 4, 5),
            (),
        ),
        (
            MultivariateGaussian(
                mu=mx.nd.zeros(shape=(3, 4, 5)),
                L=make_nd_diag(F=mx.nd, x=mx.nd.ones(shape=(3, 4, 5)), d=5),
            ),
            (3, 4),
            (5,),
        ),
        (Dirichlet(alpha=mx.nd.ones(shape=(3, 4, 5))), (3, 4), (5,)),
        (
            DirichletMultinomial(
                dim=5, n_trials=9, alpha=mx.nd.ones(shape=(3, 4, 5))
            ),
            (3, 4),
            (5,),
        ),
        (
            Laplace(
                mu=mx.nd.zeros(shape=(3, 4, 5)), b=mx.nd.ones(shape=(3, 4, 5))
            ),
            (3, 4, 5),
            (),
        ),
        (
            NegativeBinomial(
                mu=mx.nd.zeros(shape=(3, 4, 5)),
                alpha=mx.nd.ones(shape=(3, 4, 5)),
            ),
            (3, 4, 5),
            (),
        ),
        (Poisson(rate=mx.nd.ones(shape=(3, 4, 5))), (3, 4, 5), ()),
        (
            Uniform(
                low=-mx.nd.ones(shape=(3, 4, 5)),
                high=mx.nd.ones(shape=(3, 4, 5)),
            ),
            (3, 4, 5),
            (),
        ),
        (
            PiecewiseLinear(
                gamma=mx.nd.ones(shape=(3, 4, 5)),
                slopes=mx.nd.ones(shape=(3, 4, 5, 10)),
                knot_spacings=mx.nd.ones(shape=(3, 4, 5, 10)) / 10,
            ),
            (3, 4, 5),
            (),
        ),
        (
            MixtureDistribution(
                mixture_probs=mx.nd.stack(
                    0.2 * mx.nd.ones(shape=(3, 1, 5)),
                    0.8 * mx.nd.ones(shape=(3, 1, 5)),
                    axis=-1,
                ),
                components=[
                    Gaussian(
                        mu=mx.nd.zeros(shape=(3, 4, 5)),
                        sigma=mx.nd.ones(shape=(3, 4, 5)),
                    ),
                    StudentT(
                        mu=mx.nd.zeros(shape=(3, 4, 5)),
                        sigma=mx.nd.ones(shape=(3, 4, 5)),
                        nu=mx.nd.ones(shape=(3, 4, 5)),
                    ),
                ],
            ),
            (3, 4, 5),
            (),
        ),
        (
            MixtureDistribution(
                mixture_probs=mx.nd.stack(
                    0.2 * mx.nd.ones(shape=(3, 4)),
                    0.8 * mx.nd.ones(shape=(3, 4)),
                    axis=-1,
                ),
                components=[
                    MultivariateGaussian(
                        mu=mx.nd.zeros(shape=(3, 4, 5)),
                        L=make_nd_diag(
                            F=mx.nd, x=mx.nd.ones(shape=(3, 4, 5)), d=5
                        ),
                    ),
                    MultivariateGaussian(
                        mu=mx.nd.zeros(shape=(3, 4, 5)),
                        L=make_nd_diag(
                            F=mx.nd, x=mx.nd.ones(shape=(3, 4, 5)), d=5
                        ),
                    ),
                ],
            ),
            (3, 4),
            (5,),
        ),
        (
            TransformedDistribution(
                StudentT(
                    mu=mx.nd.zeros(shape=(3, 4, 5)),
                    sigma=mx.nd.ones(shape=(3, 4, 5)),
                    nu=mx.nd.ones(shape=(3, 4, 5)),
                ),
                [
                    AffineTransformation(
                        scale=1e-1 + mx.nd.random.uniform(shape=(3, 4, 5))
                    )
                ],
            ),
            (3, 4, 5),
            (),
        ),
        (
            TransformedDistribution(
                MultivariateGaussian(
                    mu=mx.nd.zeros(shape=(3, 4, 5)),
                    L=make_nd_diag(
                        F=mx.nd, x=mx.nd.ones(shape=(3, 4, 5)), d=5
                    ),
                ),
                [
                    AffineTransformation(
                        scale=1e-1 + mx.nd.random.uniform(shape=(3, 4, 5))
                    )
                ],
            ),
            (3, 4),
            (5,),
        ),
        (
            TransformedDistribution(
                Uniform(
                    low=mx.nd.zeros(shape=(3, 4, 5)),
                    high=mx.nd.ones(shape=(3, 4, 5)),
                ),
                [
                    BoxCoxTransform(
                        lambda_1=mx.nd.ones(shape=(3, 4, 5)),
                        lambda_2=mx.nd.zeros(shape=(3, 4, 5)),
                    )
                ],
            ),
            (3, 4, 5),
            (),
        ),
        (
            Loglogistic(
                mx.nd.zeros(shape=(3, 4, 5)),
                mx.nd.ones(shape=(3, 4, 5)),
            ),
            (3, 4, 5),
            (),
        ),
        (
            Weibull(
                mx.nd.ones(shape=(3, 4, 5)),
                mx.nd.ones(shape=(3, 4, 5)),
            ),
            (3, 4, 5),
            (),
        ),
    ],
)
def test_distribution_shapes(
    distr: Distribution,
    expected_batch_shape: Tuple,
    expected_event_shape: Tuple,
):
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

    def has_quantile(d):
        return isinstance(d, (Uniform, Gaussian, Laplace))

    if (
        has_quantile(distr)
        or isinstance(distr, TransformedDistribution)
        and has_quantile(distr.base_distribution)
    ):
        qs1 = distr.quantile(mx.nd.array([0.5]))
        assert qs1.shape == (1,) + distr.batch_shape + distr.event_shape
