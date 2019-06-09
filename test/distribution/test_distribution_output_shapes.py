from typing import Tuple, List, Union
import pytest

import mxnet as mx

from gluonts.model.common import Tensor
from gluonts.distribution import (
    DistributionOutput,
    GaussianOutput,
    LaplaceOutput,
    MixtureDistributionOutput,
    MultivariateGaussianOutput,
    NegativeBinomialOutput,
    PiecewiseLinearOutput,
    StudentTOutput,
    UniformOutput,
)


@pytest.mark.parametrize(
    "distr_out, data, scale, expected_batch_shape, expected_event_shape",
    [
        (
            GaussianOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            StudentTOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            MultivariateGaussianOutput(dim=5),
            mx.nd.random.normal(shape=(3, 4, 10)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4),
            (5,),
        ),
        (
            LaplaceOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            NegativeBinomialOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            UniformOutput(),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            PiecewiseLinearOutput(num_pieces=3),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
            [None, mx.nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            MixtureDistributionOutput([GaussianOutput(), StudentTOutput()]),
            mx.nd.random.normal(shape=(3, 4, 5, 6)),
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
            (3, 4),
            (5,),
        ),
    ],
)
def test_distribution_output_shapes(
    distr_out: DistributionOutput,
    data: Tensor,
    scale: List[Union[None, Tensor]],
    expected_batch_shape: Tuple,
    expected_event_shape: Tuple,
):
    args_proj = distr_out.get_args_proj()
    args_proj.initialize()

    args = args_proj(data)

    assert distr_out.event_shape == expected_event_shape

    for s in scale:

        distr = distr_out.distribution(args, scale=s)

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
