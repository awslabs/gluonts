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

from typing import List, Tuple

import mxnet as mx
import numpy as np
import pytest

from gluonts.core.serde import dump_json, load_json
from gluonts.mx.distribution import ISQF, ISQFOutput
from gluonts.testutil import empirical_cdf

serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


@pytest.mark.parametrize(
    "distr, alpha, quantile, target, left_crps, right_crps, spline_crps",
    [
        (
            ISQF(
                spline_knots=mx.nd.array([1]).reshape(shape=(1, 1, 1)),
                spline_heights=mx.nd.array([1]).reshape(shape=(1, 1, 1)),
                beta_l=mx.nd.array([2]),
                beta_r=mx.nd.array([2]),
                qk_y=mx.nd.array([0, 1]).reshape(shape=(1, 2)),
                qk_x=mx.nd.array([0.1, 0.9]).reshape(shape=(1, 2)),
                num_qk=2,
                num_pieces=1,
            ),
            [0.05, 0.3, 0.7, 0.99],
            [0.5 * np.log(0.5), 0.25, 0.75, 1 + 0.5 * np.log(10)],
            [0.5],
            [0.0075],
            [0.0075],
            [7 / 75],
        ),
        (
            ISQF(
                spline_knots=mx.nd.array([1, 1, 1, 1]).reshape(
                    shape=(1, 2, 2)
                ),
                spline_heights=mx.nd.array([1, 1, 1, 1]).reshape(
                    shape=(1, 2, 2)
                ),
                beta_l=mx.nd.array([1]),
                beta_r=mx.nd.array([1]),
                qk_y=mx.nd.array([0, 1, 2]).reshape(shape=(1, 3)),
                qk_x=mx.nd.array([0.1, 0.5, 0.9]).reshape(shape=(1, 3)),
                num_qk=3,
                num_pieces=2,
            ),
            [0.01, 0.45, 0.75, 0.95],
            [np.log(0.1), 0.875, 1.625, 2 + np.log(2)],
            [2],
            [0.025],
            [0.005],
            [44 / 75],
        ),
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
# Test the interpolation, extrapolation and crps computation
def test_values(
    distr: ISQF,
    alpha: List[float],
    quantile: List[float],
    target: List[float],
    left_crps: List[float],
    right_crps: List[float],
    spline_crps: List[float],
    serialize_fn,
):
    distr = serialize_fn(distr)

    target = mx.nd.array(target).reshape(shape=(len(target),))
    alpha = mx.nd.array(alpha).reshape((len(alpha), len(target)))
    quantile = np.array(quantile).reshape((len(quantile), len(target)))

    left_crps = np.array(left_crps)
    right_crps = np.array(right_crps)
    spline_crps = np.array(spline_crps)

    # Test the interpolation/extrapolation between the quantile knots
    assert all(np.isclose(distr.quantile(alpha).asnumpy(), quantile))

    # Test the crps computation
    assert all(
        np.isclose(
            distr.crps_tail(target, left_tail=True).asnumpy(), left_crps
        )
    )
    assert all(
        np.isclose(
            distr.crps_tail(target, left_tail=False).asnumpy(), right_crps
        )
    )
    assert all(np.isclose(distr.crps_spline(target).asnumpy(), spline_crps))
    assert all(
        np.isclose(
            distr.crps(target).asnumpy(), left_crps + right_crps + spline_crps
        )
    )


@pytest.mark.parametrize(
    "batch_shape, num_pieces, num_qk, num_samples",
    [
        ((3, 4, 5), 10, 5, 100),
        ((1,), 2, 2, 1),
        ((10,), 1, 2, 10),
        ((10, 5), 5, 5, 10),
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_shapes(
    batch_shape: Tuple,
    num_pieces: int,
    num_qk: int,
    num_samples: int,
    serialize_fn,
):

    spline_knots = mx.nd.ones(shape=(*batch_shape, (num_qk - 1), num_pieces))
    spline_heights = mx.nd.ones(shape=(*batch_shape, (num_qk - 1), num_pieces))
    beta_l = mx.nd.ones(shape=batch_shape)
    beta_r = mx.nd.ones(shape=batch_shape)

    # quantile knot positions are non-decreasing
    qk_y = mx.nd.cumsum(
        mx.nd.ones(shape=(*batch_shape, num_qk)), axis=len(batch_shape) - 1
    )
    qk_x = mx.nd.cumsum(
        mx.nd.ones(shape=(*batch_shape, num_qk)), axis=len(batch_shape) - 1
    )

    target = mx.nd.ones(shape=batch_shape)

    distr = ISQF(
        spline_knots=spline_knots,
        spline_heights=spline_heights,
        beta_l=beta_l,
        beta_r=beta_r,
        qk_y=qk_y,
        qk_x=qk_x,
        num_qk=num_qk,
        num_pieces=num_pieces,
    )
    distr = serialize_fn(distr)

    # assert that the parameters and target have proper shapes
    assert spline_knots.shape == spline_heights.shape
    assert beta_l.shape == beta_r.shape == target.shape
    assert qk_y.shape == qk_x.shape
    assert len(qk_y.shape) == len(target.shape) + 1
    assert len(spline_knots.shape) == len(target.shape) + 2

    # assert that batch_shape is computed properly
    assert distr.batch_shape == batch_shape

    # assert that shapes of original parameters are correct
    assert distr.sk_x.shape == distr.sk_y.shape == spline_knots.shape
    assert (
        distr.delta_sk_x.shape == distr.delta_sk_y.shape == spline_knots.shape
    )

    assert distr.qk_x.shape == distr.qk_y.shape == spline_knots.shape[:-1]
    assert (
        distr.qk_x_plus.shape
        == distr.qk_y_plus.shape
        == spline_knots.shape[:-1]
    )
    assert distr.qk_x_l.shape == distr.qk_y_l.shape == batch_shape
    assert distr.qk_x_r.shape == distr.qk_y_r.shape == batch_shape

    assert distr.tail_al.shape == distr.tail_bl.shape == batch_shape
    assert distr.tail_ar.shape == distr.tail_br.shape == batch_shape

    # assert that the shape of crps is correct
    assert distr.crps(target).shape == batch_shape

    # assert that the quantile shape is correct when computing
    # the quantile values at knot positions - used for alpha_tilde
    assert distr.quantile_spline(spline_knots, axis=-2).shape == (
        *batch_shape,
        num_qk - 1,
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


@pytest.mark.parametrize(
    "batch_shape, num_pieces, list_qk_x, num_samples",
    [
        ((20, 10, 30), 10, (0.1, 0.3, 0.5, 0.7, 0.9), 100),
        ((5000,), 2, (0.1, 0.9), 1),
        ((3000,), 1, (0.1, 0.9), 10),
        ((1000, 30), 5, (0.1, 0.3, 0.5, 0.7, 0.9), 10),
    ],
)
@pytest.mark.parametrize(
    "atol",
    [
        1e-2,
    ],
)
# Test quantile(cdf(y))=y and cdf(quantile(alpha))=alpha
def test_cdf_quantile_consistency(
    batch_shape: Tuple,
    num_pieces: int,
    list_qk_x: List[float],
    num_samples: int,
    atol: float,
):

    distr_out = ISQFOutput(num_pieces, list_qk_x)

    args_proj = distr_out.get_args_proj()
    args_proj.initialize()

    net_out = mx.nd.random.normal(shape=(*batch_shape, 30))
    args = args_proj(net_out)
    distr = distr_out.distribution(args)

    alpha = mx.nd.random_uniform(shape=batch_shape)
    alpha_estimate = distr.cdf(distr.quantile_internal(alpha))

    assert mx.nd.max(mx.nd.abs(alpha_estimate - alpha)) < atol

    y = mx.nd.random_normal(shape=batch_shape)
    y_approx = distr.quantile_internal(distr.cdf(y))

    assert mx.nd.max(mx.nd.abs(y_approx - y)) < atol


def test_robustness():
    distr_out = ISQFOutput(num_pieces=10, qk_x=[0.1, 0.5, 0.9])
    args_proj = distr_out.get_args_proj()
    args_proj.initialize()

    net_out = mx.nd.random.normal(shape=(1000, 30), scale=1e2)
    args = args_proj(net_out)
    distr = distr_out.distribution(args)

    margin = mx.nd.random.normal(shape=(1000,), scale=1e2)
    x = mx.random.uniform(
        low=distr.qk_y_l - margin, high=distr.qk_y_r + margin
    )

    # check that 0 < cdf < 1
    cdf_x = distr.cdf(x)
    assert (
        mx.nd.min(cdf_x).asscalar() >= 0.0
        and mx.nd.max(cdf_x).asscalar() <= 1.0
    )

    # check that 0 <= crps
    crps_x = distr.crps(x)
    assert mx.nd.min(crps_x).asscalar() >= 0.0
