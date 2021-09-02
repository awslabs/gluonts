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
from typing import NamedTuple, Optional

import mxnet as mx
import numpy as np
import pytest

import gluonts.mx.distribution.bijection as bij
from gluonts.mx.distribution import (
    Beta,
    Binned,
    Dirichlet,
    Gamma,
    Laplace,
    MixtureDistribution,
    NegativeBinomial,
    PiecewiseLinear,
    Poisson,
    StudentT,
    TransformedDistribution,
    Uniform,
)
from gluonts.mx.distribution.box_cox_transform import BoxCoxTransform
from gluonts.mx.distribution.gaussian import Gaussian


@pytest.mark.parametrize(
    "slice_axis_args, expected_axis_length",
    [[(0, 0, None), 3], [(0, 1, 3), 2], [(1, -1, None), 1]],
)
@pytest.mark.parametrize(
    "distr",
    [
        Gaussian(
            mu=mx.nd.random.normal(shape=(3, 4)),
            sigma=mx.nd.random.uniform(shape=(3, 4)),
        )
    ],
)
def test_distr_slice_axis(distr, slice_axis_args, expected_axis_length):
    axis, begin, end = slice_axis_args
    distr_sliced = distr.slice_axis(axis, begin, end)

    assert distr_sliced.batch_shape[axis] == expected_axis_length


class SliceHelper:
    def __getitem__(self, item):
        return item


sh = SliceHelper()

BATCH_SHAPE = (3, 4, 5)


DISTRIBUTIONS_WITH_QUANTILE_FUNCTION = (Gaussian, Uniform, Laplace, Binned)


@pytest.mark.parametrize(
    "distr",
    [
        TransformedDistribution(
            Gaussian(
                mu=mx.nd.random.uniform(shape=BATCH_SHAPE),
                sigma=mx.nd.ones(shape=BATCH_SHAPE),
            ),
            [
                bij.AffineTransformation(
                    scale=1e-1 + mx.nd.random.uniform(shape=BATCH_SHAPE)
                ),
                bij.softrelu,
            ],
        ),
        Binned(
            bin_log_probs=mx.nd.uniform(shape=BATCH_SHAPE + (23,)),
            bin_centers=mx.nd.array(np.logspace(-1, 1, 23))
            + mx.nd.zeros(BATCH_SHAPE + (23,)),
        ),
        TransformedDistribution(
            Binned(
                bin_log_probs=mx.nd.uniform(shape=BATCH_SHAPE + (23,)),
                bin_centers=mx.nd.array(np.logspace(-1, 1, 23))
                + mx.nd.zeros(BATCH_SHAPE + (23,)),
            ),
            [
                bij.AffineTransformation(
                    scale=1e-1 + mx.nd.random.uniform(shape=BATCH_SHAPE)
                ),
                bij.softrelu,
            ],
        ),
        Gaussian(
            mu=mx.nd.zeros(shape=BATCH_SHAPE),
            sigma=mx.nd.ones(shape=BATCH_SHAPE),
        ),
        Gamma(
            alpha=mx.nd.ones(shape=BATCH_SHAPE),
            beta=mx.nd.ones(shape=BATCH_SHAPE),
        ),
        Beta(
            alpha=0.5 * mx.nd.ones(shape=BATCH_SHAPE),
            beta=0.5 * mx.nd.ones(shape=BATCH_SHAPE),
        ),
        StudentT(
            mu=mx.nd.zeros(shape=BATCH_SHAPE),
            sigma=mx.nd.ones(shape=BATCH_SHAPE),
            nu=mx.nd.ones(shape=BATCH_SHAPE),
        ),
        Dirichlet(alpha=mx.nd.ones(shape=BATCH_SHAPE)),
        Laplace(
            mu=mx.nd.zeros(shape=BATCH_SHAPE), b=mx.nd.ones(shape=BATCH_SHAPE)
        ),
        NegativeBinomial(
            mu=mx.nd.zeros(shape=BATCH_SHAPE),
            alpha=mx.nd.ones(shape=BATCH_SHAPE),
        ),
        Poisson(rate=mx.nd.ones(shape=BATCH_SHAPE)),
        Uniform(
            low=-mx.nd.ones(shape=BATCH_SHAPE),
            high=mx.nd.ones(shape=BATCH_SHAPE),
        ),
        PiecewiseLinear(
            gamma=mx.nd.ones(shape=BATCH_SHAPE),
            slopes=mx.nd.ones(shape=(3, 4, 5, 10)),
            knot_spacings=mx.nd.ones(shape=(3, 4, 5, 10)) / 10,
        ),
        MixtureDistribution(
            mixture_probs=mx.nd.stack(
                0.2 * mx.nd.ones(shape=BATCH_SHAPE),
                0.8 * mx.nd.ones(shape=BATCH_SHAPE),
                axis=-1,
            ),
            components=[
                Gaussian(
                    mu=mx.nd.zeros(shape=BATCH_SHAPE),
                    sigma=mx.nd.ones(shape=BATCH_SHAPE),
                ),
                StudentT(
                    mu=mx.nd.zeros(shape=BATCH_SHAPE),
                    sigma=mx.nd.ones(shape=BATCH_SHAPE),
                    nu=mx.nd.ones(shape=BATCH_SHAPE),
                ),
            ],
        ),
        TransformedDistribution(
            StudentT(
                mu=mx.nd.zeros(shape=BATCH_SHAPE),
                sigma=mx.nd.ones(shape=BATCH_SHAPE),
                nu=mx.nd.ones(shape=BATCH_SHAPE),
            ),
            [
                bij.AffineTransformation(
                    scale=1e-1 + mx.nd.random.uniform(shape=BATCH_SHAPE)
                )
            ],
        ),
        TransformedDistribution(
            Uniform(
                low=mx.nd.zeros(shape=BATCH_SHAPE),
                high=mx.nd.ones(shape=BATCH_SHAPE),
            ),
            [
                BoxCoxTransform(
                    lambda_1=mx.nd.ones(shape=BATCH_SHAPE),
                    lambda_2=mx.nd.zeros(shape=BATCH_SHAPE),
                )
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "slice_item", [sh[1:2], sh[1, :], sh[:, 0], sh[0, -1]]
)
def test_slice_axis_results(distr, slice_item):
    s = distr.sample().asnumpy()
    sliced = distr[slice_item]
    s_sliced = sliced.sample().asnumpy()
    assert s_sliced.shape == s[slice_item].shape

    y = np.random.uniform(size=BATCH_SHAPE)
    lp_expected = distr.loss(mx.nd.array(y)).asnumpy()[slice_item]
    lp_actual = sliced.loss(mx.nd.array(y[slice_item])).asnumpy()
    assert np.allclose(lp_actual, lp_expected)

    tmp = (
        distr.base_distribution
        if isinstance(distr, TransformedDistribution)
        else distr
    )
    has_quantile_fn = isinstance(tmp, DISTRIBUTIONS_WITH_QUANTILE_FUNCTION)

    if has_quantile_fn:
        for ql in [0.01, 0.1, 0.5, 0.9, 0.99]:
            qs_actual = sliced.quantile(mx.nd.array([ql])).asnumpy()[0]
            qs_expected = distr.quantile(mx.nd.array([ql])).asnumpy()[0][
                slice_item
            ]
            assert np.allclose(qs_actual, qs_expected)
