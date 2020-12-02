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

import mxnet as mx

import numpy as np

from gluonts.mx.distribution.gaussian import Gaussian
from gluonts.mx.distribution.mixture import MixtureDistribution
from gluonts.mx.distribution.uniform import Uniform

# In these tests we can use only distributions that have a cdf method but do not have a quantile method.
# The mixture distribution satisfies these conditions. The components of the mixture should have a cdf method.


def test_quantile() -> None:
    r"""
    Tests if the quantiles of a single Gaussian and the quantiles of the mixture of two Gaussians
    identical to the first are equal. The quantiles of the single Gaussian are given by the
    Gaussian.quantile() method while the quantiles of the mixture from the Distribution.quantile() method.
    """
    mu = mx.nd.array(
        [[1, 10, 100, 1000, 10000], [-1, -10, -100, -1000, -10000]]
    )
    sigma = mx.nd.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 2)

    gau = Gaussian(mu, sigma)

    mixture_probs = mx.nd.broadcast_like(
        mx.nd.array([[0.5, 0.5]]).expand_dims(axis=0),
        mu,
        lhs_axes=(0, 1),
        rhs_axes=(0, 1),
    )
    mix = MixtureDistribution(
        mixture_probs=mixture_probs,
        components=[Gaussian(mu, sigma), Gaussian(mu, sigma)],
    )

    quantiles = mx.nd.array([0.1, 0.5, 0.9])
    gau_quantiles = gau.quantile(quantiles)
    mix_quantiles = mix.quantile(quantiles)

    relative_error = np.max(
        np.abs(gau_quantiles.asnumpy() / mix_quantiles.asnumpy() - 1)
    )
    assert relative_error < 1e-4


def test_inverse_quantile() -> None:
    r"""
    Tests CDF(Q(x)) == x.
    """
    mu = mx.nd.array([1])
    sigma = mx.nd.array([0.1])
    uni_bound = mx.nd.array([1])
    mixture_probs = mx.nd.array([[0.3, 0.7]])

    levels = mx.nd.array([0.01, 0.1, 0.5, 0.9, 0.99])

    mix_gau = MixtureDistribution(
        mixture_probs=mixture_probs,
        components=[Gaussian(mu, sigma), Gaussian(mu, sigma)],
    )
    mix_uni = MixtureDistribution(
        mixture_probs=mixture_probs,
        components=[
            Uniform(uni_bound, 3 * uni_bound),
            Uniform(uni_bound, 3 * uni_bound),
        ],
    )
    mix_gau_uni = MixtureDistribution(
        mixture_probs=mixture_probs,
        components=[Gaussian(mu, sigma), Uniform(uni_bound, 3 * uni_bound)],
    )

    gau_res = mix_gau.cdf(mix_gau.quantile(levels))
    uni_res = mix_uni.cdf(mix_uni.quantile(levels))
    gau_uni_res = mix_gau_uni.cdf(mix_gau_uni.quantile(levels))

    assert (
        np.max(gau_res.asnumpy() / levels.asnumpy().reshape(-1, 1) - 1) < 1e-4
    )
    assert (
        np.max(uni_res.asnumpy() / levels.asnumpy().reshape(-1, 1) - 1) < 1e-4
    )
    assert (
        np.max(gau_uni_res.asnumpy() / levels.asnumpy().reshape(-1, 1) - 1)
        < 1e-4
    )
