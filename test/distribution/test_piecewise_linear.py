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

from typing import Tuple, List
import pytest

import mxnet as mx
import numpy as np

from gluonts.distribution import PiecewiseLinear, PiecewiseLinearOutput
from gluonts.testutil import empirical_cdf
from gluonts.core.serde import dump_json, load_json

serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


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
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_values(
    distr: PiecewiseLinear,
    target: List[float],
    expected_target_cdf: List[float],
    expected_target_crps: List[float],
    serialize_fn,
):
    distr = serialize_fn(distr)
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
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_shapes(
    batch_shape: Tuple, num_pieces: int, num_samples: int, serialize_fn
):
    gamma = mx.nd.ones(shape=(*batch_shape,))
    slopes = mx.nd.ones(shape=(*batch_shape, num_pieces))  # all positive
    knot_spacings = (
        mx.nd.ones(shape=(*batch_shape, num_pieces)) / num_pieces
    )  # positive and sum to 1
    target = mx.nd.ones(shape=batch_shape)  # shape of gamma

    distr = PiecewiseLinear(
        gamma=gamma, slopes=slopes, knot_spacings=knot_spacings
    )
    distr = serialize_fn(distr)

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
    assert distr.quantile_internal(knot_spacings, axis=-2).shape == (
        *batch_shape,
        num_pieces,
    )

    # assert that the samples and the quantile values shape when num_samples is None is correct
    samples = distr.sample()
    assert samples.shape == batch_shape
    assert distr.quantile_internal(samples).shape == batch_shape

    # assert that the samples and the quantile values shape when num_samples is not None is correct
    samples = distr.sample(num_samples)
    assert samples.shape == (num_samples, *batch_shape)
    assert distr.quantile_internal(samples, axis=0).shape == (
        num_samples,
        *batch_shape,
    )


def test_simple_symmetric():
    gamma = mx.nd.array([-1.0])
    slopes = mx.nd.array([[2.0, 2.0]])
    knot_spacings = mx.nd.array([[0.5, 0.5]])

    distr = PiecewiseLinear(
        gamma=gamma, slopes=slopes, knot_spacings=knot_spacings
    )

    assert distr.cdf(mx.nd.array([-2.0])).asnumpy().item() == 0.0
    assert distr.cdf(mx.nd.array([+2.0])).asnumpy().item() == 1.0

    expected_crps = np.array([1.0 + 2.0 / 3.0])

    assert np.allclose(
        distr.crps(mx.nd.array([-2.0])).asnumpy(), expected_crps
    )

    assert np.allclose(distr.crps(mx.nd.array([2.0])).asnumpy(), expected_crps)


def test_robustness():
    distr_out = PiecewiseLinearOutput(num_pieces=10)
    args_proj = distr_out.get_args_proj()
    args_proj.initialize()

    net_out = mx.nd.random.normal(shape=(1000, 30), scale=1e2)
    gamma, slopes, knot_spacings = args_proj(net_out)
    distr = distr_out.distribution((gamma, slopes, knot_spacings))

    # compute the 1-quantile (the 0-quantile is gamma)
    sup_support = gamma + mx.nd.sum(slopes * knot_spacings, axis=-1)

    assert mx.nd.broadcast_lesser_equal(gamma, sup_support).asnumpy().all()

    width = sup_support - gamma
    x = mx.random.uniform(low=gamma - width, high=sup_support + width)

    # check that 0 < cdf < 1
    cdf_x = distr.cdf(x)
    assert (
        mx.nd.min(cdf_x).asscalar() >= 0.0
        and mx.nd.max(cdf_x).asscalar() <= 1.0
    )

    # check that 0 <= crps
    crps_x = distr.crps(x)
    assert mx.nd.min(crps_x).asscalar() >= 0.0
