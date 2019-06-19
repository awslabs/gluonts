from typing import Tuple, List
import pytest

import mxnet as mx
import numpy as np

from gluonts.distribution import PiecewiseLinear
from gluonts.testutil import empirical_cdf


@pytest.mark.parametrize(
    "distr, target, expected_target_cdf, expected_target_crps",
    [
        (
            PiecewiseLinear(
                gamma=mx.nd.ones(shape=(1,)),
                slopes=mx.nd.array([2, 3, 1]).reshape(shape=(1, 3)),
                knot_spacings=mx.nd.array([0.3, 0.4, 0.3]).reshape(
                    shape=(1, 3)
                ),
            ),
            [2.2],
            [0.5],
            [0.223000],
        ),
        (
            PiecewiseLinear(
                gamma=mx.nd.ones(shape=(2,)),
                slopes=mx.nd.array([[1, 1], [1, 2]]).reshape(shape=(2, 2)),
                knot_spacings=mx.nd.array([[0.4, 0.6], [0.4, 0.6]]).reshape(
                    shape=(2, 2)
                ),
            ),
            [1.5, 1.6],
            [0.5, 0.5],
            [0.083333, 0.145333],
        ),
    ],
)
def test_values(
    distr: PiecewiseLinear,
    target: List[float],
    expected_target_cdf: List[float],
    expected_target_crps: List[float],
):
    target = mx.nd.array(target).reshape(shape=(len(target),))
    expected_target_cdf = np.array(expected_target_cdf).reshape(
        (len(expected_target_cdf),)
    )
    expected_target_crps = np.array(expected_target_crps).reshape(
        (len(expected_target_crps),)
    )

    assert all(np.isclose(distr.cdf(target).asnumpy(), expected_target_cdf))
    assert all(np.isclose(distr.crps(target).asnumpy(), expected_target_crps))

    # compare with empirical cdf from samples
    num_samples = 100_000
    samples = distr.sample(num_samples).asnumpy()
    assert np.isfinite(samples).all()

    emp_cdf, edges = empirical_cdf(samples)
    calc_cdf = distr.cdf(mx.nd.array(edges)).asnumpy()
    assert np.allclose(calc_cdf[1:, :], emp_cdf, atol=1e-2)


@pytest.mark.parametrize(
    "batch_shape, num_pieces, num_samples",
    [((3, 4, 5), 10, 100), ((1,), 2, 1), ((10,), 10, 10), ((10, 5), 2, 1)],
)
def test_shapes(batch_shape: Tuple, num_pieces: int, num_samples: int):
    gamma = mx.nd.ones(shape=(*batch_shape,))
    slopes = mx.nd.ones(shape=(*batch_shape, num_pieces))  # all positive
    knot_spacings = (
        mx.nd.ones(shape=(*batch_shape, num_pieces)) / num_pieces
    )  # positive and sum to 1
    target = mx.nd.ones(shape=batch_shape)  # shape of gamma

    distr = PiecewiseLinear(
        gamma=gamma, slopes=slopes, knot_spacings=knot_spacings
    )

    # assert that the parameters and target have proper shapes
    assert gamma.shape == target.shape
    assert knot_spacings.shape == slopes.shape
    assert len(gamma.shape) + 1 == len(knot_spacings.shape)

    # assert that batch_shape is computed properly
    assert distr.batch_shape == batch_shape

    # assert that shapes of original parameters are correct
    assert distr.b.shape == slopes.shape
    assert distr.knot_positions.shape == knot_spacings.shape

    # assert that the shape of crps is correct
    assert distr.crps(target).shape == batch_shape

    # assert that the quantile shape is correct when computing the quantile values at knot positions - used for a_tilde
    assert distr.quantile(knot_spacings, axis=-2).shape == (
        *batch_shape,
        num_pieces,
    )

    # assert that the samples and the quantile values shape when num_samples is None is correct
    samples = distr.sample()
    assert samples.shape == batch_shape
    assert distr.quantile(samples).shape == batch_shape

    # assert that the samples and the quantile values shape when num_samples is not None is correct
    samples = distr.sample(num_samples)
    assert samples.shape == (num_samples, *batch_shape)
    assert distr.quantile(samples, axis=0).shape == (num_samples, *batch_shape)
