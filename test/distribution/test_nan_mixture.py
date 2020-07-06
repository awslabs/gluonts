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

# Third-party imports
import mxnet as mx
import numpy as np
import pytest

# First-party imports
from gluonts.gluonts_tqdm import tqdm
from gluonts.model.common import Tensor, NPArrayLike
from gluonts.mx.distribution.distribution import Distribution
from gluonts.mx.distribution import (
    Gaussian,
    NanMixture,
    GaussianOutput,
    StudentTOutput,
    LaplaceOutput,
    MultivariateGaussianOutput,
)
from gluonts.testutil import empirical_cdf
from gluonts.core.serde import dump_json, load_json

serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


def diff(x: NPArrayLike, y: NPArrayLike) -> np.ndarray:
    return np.mean(np.abs(x - y))


NUM_SAMPLES = 1_000
NUM_SAMPLES_LARGE = 100_000

p = np.array([[[0.0, 0.0, 0.5]], [[1.0, 1.0, 0.5]]])
SHAPE = p.shape


@pytest.mark.parametrize(
    "distr, p",
    [
        (
            Gaussian(
                mu=mx.nd.zeros(shape=SHAPE),
                sigma=1e-3 + mx.nd.ones(shape=SHAPE),
            ),
            mx.nd.array(p),
        ),
        # TODO: add different base distributions
        # TODO: add a multivariate case here
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_nan_mixture(distr: Distribution, p: Tensor, serialize_fn) -> None:
    # sample from component distributions, and select samples

    samples = distr.sample(num_samples=NUM_SAMPLES_LARGE)

    rand = mx.nd.random.uniform(shape=(NUM_SAMPLES_LARGE, *p.shape))
    choice = (rand > p.expand_dims(axis=0)).broadcast_like(samples)
    samples_ref = mx.nd.where(choice, samples, samples.zeros_like() / 0.0)

    # construct NanMixture distribution and sample from it
    nan_mixture = NanMixture(nan_prob=p, distribution=distr)
    nan_mixture = serialize_fn(nan_mixture)

    samples_mix = nan_mixture.sample(num_samples=NUM_SAMPLES_LARGE)

    # check that shapes are right

    assert samples.shape == samples_mix.shape == samples_ref.shape
    # TODO check histogram of non-nan values
    # TODO check mean and stddev

    x = mx.nd.array([[[np.nan, 10.5, -0.5]], [[np.nan, 10.5, np.nan]]])
    print("p")
    print(p)
    print()
    print("x")
    print(x)
    # check log_prob
    log_prob = nan_mixture.log_prob(x)
    log_prob_true = mx.nd.log(mx.nd.where(x != x, p, (1 - p) * distr.prob(x)))

    assert np.allclose(log_prob.asnumpy(), log_prob_true.asnumpy(), atol=1e-1)

    # check gradients
    mu = mx.nd.zeros(shape=SHAPE)
    sigma = 1e-3 + mx.nd.ones(shape=SHAPE)

    p.attach_grad()
    mu.attach_grad()
    sigma.attach_grad()

    with mx.autograd.record():
        distr = Gaussian(mu=mu, sigma=sigma,)
        nan_mixture = NanMixture(nan_prob=p, distribution=distr)
        nll = nan_mixture.loss(x)
    nll.backward()

    p_grad_true = mx.nd.where(x != x, -1 / p, 1 / (1 - p))

    mu_grad_true = -(x - mu) / mx.nd.square(sigma)
    mu_grad_true = mx.nd.where(x != x, mu.zeros_like(), mu_grad_true)
    mu_grad_true = mx.nd.where(p == 1, mu.zeros_like(), mu_grad_true)

    sigma_grad_true = -(
        mx.nd.square(mu) - 2 * mu * x - mx.nd.square(sigma) + mx.nd.square(x)
    ) / (sigma ** 3)
    sigma_grad_true = mx.nd.where(x != x, sigma.zeros_like(), sigma_grad_true)
    sigma_grad_true = mx.nd.where(p == 1, sigma.zeros_like(), sigma_grad_true)

    print(p.grad)
    print(p_grad_true)
    print(mu.grad)
    print(mu_grad_true)
    print(sigma.grad)
    print(sigma_grad_true)
    assert np.allclose(p.grad, p_grad_true, atol=1e-1)
    assert np.allclose(mu.grad, mu_grad_true, atol=1e-1)
    assert np.allclose(sigma.grad, p_grad_true, atol=1e-1)
