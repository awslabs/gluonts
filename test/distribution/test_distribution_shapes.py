from typing import Tuple
import pytest

import mxnet as mx

from gluonts.support.util import make_nd_diag
from gluonts.distribution import (
    Distribution,
    Gaussian,
    Laplace,
    MixtureDistribution,
    MultivariateGaussian,
    NegativeBinomial,
    PiecewiseLinear,
    StudentT,
    Uniform,
    TransformedDistribution,
)
from gluonts.distribution.bijection import AffineTransformation
from gluonts.distribution.box_cox_tranform import BoxCoxTranform


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
                AffineTransformation(
                    scale=1e-1 + mx.nd.random.uniform(shape=(3, 4, 5))
                ),
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
                AffineTransformation(
                    scale=1e-1 + mx.nd.random.uniform(shape=(3, 4, 5))
                ),
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
                BoxCoxTranform(
                    lambda_1=mx.nd.ones(shape=(3, 4, 5)),
                    lambda_2=mx.nd.zeros(shape=(3, 4, 5)),
                ),
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
