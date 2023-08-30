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

import torch
import numpy as np

from gluonts.torch.distributions import (
    PiecewiseLinear,
    PiecewiseLinearOutput,
)


def empirical_cdf(
    samples: np.ndarray, num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empirical cdf from the given samples.
    Parameters
    ----------
    samples
        Tensor of samples of shape (num_samples, batch_shape)
    Returns
    -------
    Tensor
        Empirically calculated cdf values. shape (num_bins, batch_shape)
    Tensor
        Bin edges corresponding to the cdf values. shape (num_bins + 1, batch_shape)
    """

    # calculate histogram separately for each dimension in the batch size
    cdfs = []
    edges = []

    batch_shape = samples.shape[1:]
    agg_batch_dim = np.prod(batch_shape, dtype=int)

    samples = samples.reshape((samples.shape[0], -1))

    for i in range(agg_batch_dim):
        s = samples[:, i]
        bins = np.linspace(s.min(), s.max(), num_bins + 1)
        hist, edge = np.histogram(s, bins=bins)
        cdfs.append(np.cumsum(hist / len(s)))
        edges.append(edge)

    empirical_cdf = np.stack(cdfs, axis=-1).reshape(num_bins, *batch_shape)
    edges = np.stack(edges, axis=-1).reshape(num_bins + 1, *batch_shape)
    return empirical_cdf, edges


@pytest.mark.parametrize(
    "distr, target, expected_target_cdf, expected_target_crps",
    [
        (
            PiecewiseLinear(
                gamma=torch.ones(size=(1,)),
                slopes=torch.Tensor([2, 3, 1]).reshape(shape=(1, 3)),
                knot_spacings=torch.Tensor([0.3, 0.4, 0.3]).reshape(
                    shape=(1, 3)
                ),
            ),
            [2.2],
            [0.5],
            [0.223000],
        ),
        (
            PiecewiseLinear(
                gamma=torch.ones(size=(2,)),
                slopes=torch.Tensor([[1, 1], [1, 2]]).reshape(shape=(2, 2)),
                knot_spacings=torch.Tensor([[0.4, 0.6], [0.4, 0.6]]).reshape(
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
    target = torch.Tensor(target).reshape(shape=(len(target),))
    expected_target_cdf = np.array(expected_target_cdf).reshape(
        (len(expected_target_cdf),)
    )
    expected_target_crps = np.array(expected_target_crps).reshape(
        (len(expected_target_crps),)
    )

    assert all(np.isclose(distr.cdf(target).numpy(), expected_target_cdf))
    assert all(np.isclose(distr.crps(target).numpy(), expected_target_crps))

    # compare with empirical cdf from samples
    num_samples = 100_000
    samples = distr.sample((num_samples,)).numpy()
    assert np.isfinite(samples).all()

    emp_cdf, edges = empirical_cdf(samples)
    calc_cdf = distr.cdf(torch.Tensor(edges)).numpy()
    assert np.allclose(calc_cdf[1:, :], emp_cdf, atol=1e-2)


@pytest.mark.parametrize(
    "batch_shape, num_pieces, num_samples",
    [((3, 4, 5), 10, 100), ((1,), 2, 1), ((10,), 10, 10), ((10, 5), 2, 1)],
)
def test_shapes(batch_shape: Tuple, num_pieces: int, num_samples: int):
    gamma = torch.ones(size=(*batch_shape,))
    slopes = torch.ones(size=(*batch_shape, num_pieces))  # all positive
    knot_spacings = (
        torch.ones(size=(*batch_shape, num_pieces)) / num_pieces
    )  # positive and sum to 1
    target = torch.ones(size=batch_shape)  # shape of gamma

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

    # assert that the quantile shape is correct when computing the
    # quantile values at knot positions - used for a_tilde
    assert distr.quantile_internal(knot_spacings, dim=-2).shape == (
        *batch_shape,
        num_pieces,
    )

    # assert that the samples and the quantile values shape when num_samples
    # is None is correct
    samples = distr.sample()
    assert samples.shape == batch_shape
    assert distr.quantile_internal(samples).shape == batch_shape

    # assert that the samples and the quantile values shape when num_samples
    # is not None is correct
    samples = distr.sample((num_samples,))
    assert samples.shape == (num_samples, *batch_shape)
    assert distr.quantile_internal(samples, dim=0).shape == (
        num_samples,
        *batch_shape,
    )


def test_simple_symmetric():
    gamma = torch.Tensor([-1.0])
    slopes = torch.Tensor([[2.0, 2.0]])
    knot_spacings = torch.Tensor([[0.5, 0.5]])

    distr = PiecewiseLinear(
        gamma=gamma, slopes=slopes, knot_spacings=knot_spacings
    )

    assert distr.cdf(torch.Tensor([-2.0])).numpy().item() == 0.0
    assert distr.cdf(torch.Tensor([+2.0])).numpy().item() == 1.0

    expected_crps = np.array([1.0 + 2.0 / 3.0])

    assert np.allclose(distr.crps(torch.Tensor([-2.0])).numpy(), expected_crps)

    assert np.allclose(distr.crps(torch.Tensor([2.0])).numpy(), expected_crps)


def test_robustness():
    distr_out = PiecewiseLinearOutput(num_pieces=10)
    args_proj = distr_out.get_args_proj(in_features=30)

    net_out = torch.normal(mean=0.0, size=(1000, 30), std=1e2)
    gamma, slopes, knot_spacings = args_proj(net_out)
    distr = distr_out.distribution((gamma, slopes, knot_spacings))

    # compute the 1-quantile (the 0-quantile is gamma)
    sup_support = gamma + (slopes * knot_spacings).sum(-1)

    assert torch.le(gamma, sup_support).numpy().all()

    width = sup_support - gamma
    x = torch.from_numpy(
        np.random.uniform(
            low=(gamma - width).detach().numpy(),
            high=(sup_support + width).detach().numpy(),
        ).astype(np.float32),
    )

    # check that 0 < cdf < 1
    cdf_x = distr.cdf(x)
    assert torch.min(cdf_x).item() >= 0.0 and torch.max(cdf_x).item() <= 1.0

    # check that 0 <= crps
    crps_x = distr.crps(x)
    assert torch.min(crps_x).item() >= 0.0
