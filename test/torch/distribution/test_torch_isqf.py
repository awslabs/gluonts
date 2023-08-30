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

import torch
import numpy as np
import pytest

from gluonts.torch.distributions import (
    ISQF,
    ISQFOutput,
)


@pytest.mark.parametrize(
    "distr, alpha, quantile, target, left_crps, right_crps, spline_crps",
    [
        (
            ISQF(
                spline_knots=torch.tensor([1], dtype=torch.float32).reshape(
                    1, 1, 1
                ),
                spline_heights=torch.tensor([1], dtype=torch.float32).reshape(
                    1, 1, 1
                ),
                beta_l=torch.tensor([2], dtype=torch.float32),
                beta_r=torch.tensor([2], dtype=torch.float32),
                qk_y=torch.tensor([0, 1], dtype=torch.float32).reshape(1, 2),
                qk_x=torch.tensor([0.1, 0.9], dtype=torch.float32).reshape(
                    1, 2
                ),
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
                spline_knots=torch.tensor(
                    [1, 1, 1, 1], dtype=torch.float32
                ).reshape(1, 2, 2),
                spline_heights=torch.tensor(
                    [1, 1, 1, 1], dtype=torch.float32
                ).reshape(1, 2, 2),
                beta_l=torch.tensor([1], dtype=torch.float32),
                beta_r=torch.tensor([1], dtype=torch.float32),
                qk_y=torch.tensor([0, 1, 2], dtype=torch.float32).reshape(
                    1, 3
                ),
                qk_x=torch.tensor(
                    [0.1, 0.5, 0.9], dtype=torch.float32
                ).reshape(1, 3),
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
# Test the interpolation, extrapolation and crps computation
def test_values(
    distr: ISQF,
    alpha: List[float],
    quantile: List[float],
    target: List[float],
    left_crps: List[float],
    right_crps: List[float],
    spline_crps: List[float],
):
    target = torch.tensor(target).reshape(len(target))
    alpha = torch.tensor(alpha).reshape(len(alpha), len(target))
    quantile = np.array(quantile).reshape((len(quantile), len(target)))

    left_crps = np.array(left_crps)
    right_crps = np.array(right_crps)
    spline_crps = np.array(spline_crps)

    # Test the interpolation/extrapolation between the quantile knots
    assert all(np.isclose(distr.quantile(alpha).numpy(), quantile))

    # Test the crps computation
    assert all(
        np.isclose(distr.crps_tail(target, left_tail=True).numpy(), left_crps)
    )
    assert all(
        np.isclose(
            distr.crps_tail(target, left_tail=False).numpy(), right_crps
        )
    )
    assert all(np.isclose(distr.crps_spline(target).numpy(), spline_crps))
    assert all(
        np.isclose(
            distr.crps(target).numpy(), left_crps + right_crps + spline_crps
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
def test_shapes(
    batch_shape: Tuple,
    num_pieces: int,
    num_qk: int,
    num_samples: int,
):
    spline_knots = torch.ones(
        (*batch_shape, (num_qk - 1), num_pieces), dtype=torch.float32
    )
    spline_heights = torch.ones(
        (*batch_shape, (num_qk - 1), num_pieces), dtype=torch.float32
    )
    beta_l = torch.ones(batch_shape, dtype=torch.float32)
    beta_r = torch.ones(batch_shape, dtype=torch.float32)

    # quantile knot positions are non-decreasing
    qk_y = torch.cumsum(
        torch.ones((*batch_shape, num_qk), dtype=torch.float32), dim=-1
    )
    qk_x = torch.cumsum(
        torch.ones((*batch_shape, num_qk), dtype=torch.float32), dim=-1
    )

    target = torch.ones(batch_shape, dtype=torch.float32)

    distr = ISQF(
        spline_knots=spline_knots,
        spline_heights=spline_heights,
        beta_l=beta_l,
        beta_r=beta_r,
        qk_y=qk_y,
        qk_x=qk_x,
    )

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
    assert distr.quantile_spline(spline_knots, dim=-2).shape == (
        *batch_shape,
        num_qk - 1,
        num_pieces,
    )

    # assert that the samples and the quantile values shape when num_samples is None is correct
    samples = distr.sample()
    assert samples.shape == batch_shape
    assert distr.quantile_internal(samples).shape == batch_shape

    # assert that the samples and the quantile values shape when num_samples is not None is correct
    samples = distr.sample((num_samples,))
    assert samples.shape == (num_samples, *batch_shape)
    assert distr.quantile_internal(samples, dim=0).shape == (
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

    args_proj = distr_out.get_args_proj(in_features=30)

    net_out = torch.normal(mean=0.0, size=(*batch_shape, 30), std=1.0)
    args = args_proj(net_out)
    distr = distr_out.distribution(args)

    alpha = torch.rand(size=batch_shape)
    alpha_estimate = distr.cdf(distr.quantile_internal(alpha))

    assert torch.max(torch.abs(alpha_estimate - alpha)) < atol

    y = torch.normal(mean=0.0, size=batch_shape, std=1.0)
    y_approx = distr.quantile_internal(distr.cdf(y))

    assert torch.max(torch.abs(y_approx - y)) < atol


def test_robustness():
    distr_out = ISQFOutput(num_pieces=10, qk_x=[0.1, 0.5, 0.9])
    args_proj = distr_out.get_args_proj(in_features=30)

    net_out = torch.normal(mean=0.0, size=(1000, 30), std=1e2)
    args = args_proj(net_out)
    distr = distr_out.distribution(args)

    margin = torch.normal(mean=0.0, size=(1000,), std=1e2)

    # x is uniformly random in [distr.qk_y_l - margin, distr.qk_y_r + margin]
    x = torch.rand_like(distr.qk_y_l) * (
        distr.qk_y_r - distr.qk_y_l + 2 * margin
    ) + (distr.qk_y_l - margin)

    # check that 0 < cdf < 1
    cdf_x = distr.cdf(x)
    assert torch.min(cdf_x).item() >= 0.0 and torch.max(cdf_x).item() >= 1.0

    # check that 0 <= crps
    crps_x = distr.crps(x)
    assert torch.min(crps_x).item() >= 0.0
