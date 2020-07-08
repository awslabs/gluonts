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

# Standard library imports
from typing import Iterable, List, Tuple

# Third-party imports
import mxnet as mx
import numpy as np
import pytest

# First-party imports
from gluonts.gluonts_tqdm import tqdm
from gluonts.model.common import Tensor, NPArrayLike
from gluonts.mx.distribution.distribution import Distribution

from gluonts.mx.distribution import (
    Categorical,
    CategoricalOutput,
    Gaussian,
    GaussianOutput,
    NanMixture,
    NanMixtureOutput,
    StudentTOutput,
)
from gluonts.core.serde import dump_json, load_json

serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


def diff(x: NPArrayLike, y: NPArrayLike) -> np.ndarray:
    return np.mean(np.abs(x - y))


NUM_SAMPLES = 2_000
NUM_SAMPLES_LARGE = 100_000

p = np.array([[[0.0001, 0.0001, 0.5]], [[0.999, 0.999, 0.5]]])
p_cat = np.array(
    [
        [[[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]]],
        [[[0.9, 0.1], [0.01, 0.99], [0.45, 0.55]]],
    ]
)
SHAPE = p.shape


@pytest.mark.parametrize(
    "distr, cat_distr, p, p_cat",
    [
        (
            Gaussian(
                mu=mx.nd.zeros(shape=SHAPE),
                sigma=1e-3 + mx.nd.ones(shape=SHAPE),
            ),
            Categorical(log_probs=mx.nd.log(mx.nd.array(p_cat))),
            mx.nd.array(p),
            mx.nd.array(p_cat),
        ),
        # TODO: add a multivariate case here
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_nan_mixture(
    distr: Distribution,
    cat_distr: Distribution,
    p: Tensor,
    p_cat: Tensor,
    serialize_fn,
) -> None:
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

    # TODO check mean and stddev

    x = mx.nd.array([[[np.nan, 3.5, -0.5]], [[np.nan, 3.5, np.nan]]])

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
        nll = -nan_mixture.log_prob(x)
    nll.backward()

    p_grad_true = mx.nd.where(x != x, -1 / p, 1 / (1 - p))
    # gradient is undefined for these cases:
    p_grad_true = mx.nd.where(
        mx.nd.logical_or(
            mx.nd.logical_and(x != x, p == 0),
            mx.nd.logical_and(x == x, p == 1),
        ),
        0.0 / p_grad_true.zeros_like(),
        p_grad_true,
    )

    mu_grad_true = -(x - mu) / mx.nd.square(sigma)
    mu_grad_true = mx.nd.where(x != x, mu.zeros_like(), mu_grad_true)
    mu_grad_true = mx.nd.where(p == 1, mu.zeros_like(), mu_grad_true)

    sigma_grad_true = -(
        mx.nd.square(mu) - 2 * mu * x - mx.nd.square(sigma) + mx.nd.square(x)
    ) / (sigma ** 3)
    sigma_grad_true = mx.nd.where(x != x, sigma.zeros_like(), sigma_grad_true)
    sigma_grad_true = mx.nd.where(p == 1, sigma.zeros_like(), sigma_grad_true)

    assert np.allclose(p.grad.asnumpy(), p_grad_true.asnumpy(), atol=1e-1)
    assert np.allclose(mu.grad.asnumpy(), mu_grad_true.asnumpy(), atol=1e-1)
    assert np.allclose(
        sigma.grad.asnumpy(), sigma_grad_true.asnumpy(), atol=1e-1
    )

    # Check if NanMixture works with a discrete distribution
    x = mx.nd.array([[[np.nan, 0, 1]], [[np.nan, 0, np.nan]]])

    # construct NanMixture distribution
    nan_mixture = NanMixture(nan_prob=p, distribution=cat_distr)
    nan_mixture = serialize_fn(nan_mixture)

    # check log_prob
    log_prob = nan_mixture.log_prob(x)
    log_prob_true = mx.nd.log(
        mx.nd.where(x != x, p, (1 - p) * cat_distr.prob(x))
    )

    assert np.allclose(log_prob.asnumpy(), log_prob_true.asnumpy(), atol=1e-1)

    # check the gradient

    p_cat.attach_grad()
    p.attach_grad()

    with mx.autograd.record():
        cat_distr = Categorical(log_probs=mx.nd.log(p_cat))
        nan_mixture = NanMixture(nan_prob=p, distribution=cat_distr)
        nll = -nan_mixture.log_prob(x)
    nll.backward()

    p_grad_true = mx.nd.where(x != x, -1 / p, 1 / (1 - p))

    p_cat_zero = p_cat[:, :, :, 0]
    p_cat_one = p_cat[:, :, :, 1]

    p_cat_grad_zero_true = mx.nd.where(
        mx.nd.logical_or(x != x, x == 1),
        mx.nd.array(p_cat_zero).zeros_like(),
        -1 / mx.nd.array(p_cat_zero),
    )
    p_cat_grad_one_true = mx.nd.where(
        mx.nd.logical_or(x != x, x == 0),
        mx.nd.array(p_cat_one).zeros_like(),
        -1 / mx.nd.array(p_cat_one),
    )
    p_cat_grad_true = mx.nd.stack(
        p_cat_grad_zero_true, p_cat_grad_one_true, axis=-1
    )

    assert np.allclose(p.grad.asnumpy(), p_grad_true.asnumpy(), atol=1e-1)
    assert np.allclose(
        p_cat.grad.asnumpy(), p_cat_grad_true.asnumpy(), atol=1e-1
    )


@pytest.mark.parametrize(
    "distribution_output", [GaussianOutput(), StudentTOutput(),],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_nanmixture_output(distribution_output, serialize_fn) -> None:

    nmdo = NanMixtureOutput(distribution_output)

    args_proj = nmdo.get_args_proj()
    args_proj.initialize()

    input = mx.nd.ones(shape=(3, 2))

    distr_args = args_proj(input)

    d = nmdo.distribution(distr_args)
    print(d)
    d = serialize_fn(d)

    samples = d.sample(num_samples=NUM_SAMPLES)

    sample = d.sample()

    assert samples.shape == (NUM_SAMPLES, *sample.shape)

    log_prob = d.log_prob(sample)

    assert log_prob.shape == d.batch_shape


BATCH_SIZE = 10000

zeros = mx.nd.zeros((BATCH_SIZE))
ones = mx.nd.ones((BATCH_SIZE))

mu = 1.0
sigma = 1.5
TOL = 0.09

nan_prob = 0.3
mx.random.seed(1)
np.random.seed(1)

samples = np.random.normal(mu, scale=sigma, size=(BATCH_SIZE))
np_samples = np.where(
    np.random.uniform(size=(BATCH_SIZE)) > nan_prob, samples, np.nan
)

n_cat = 3
cat_probs = np.array([0.2, 0.3, 0.5])
cat_samples = np.random.choice(
    list(range(n_cat)), p=cat_probs, size=BATCH_SIZE
)
cat_samples = np.where(
    np.random.uniform(size=(BATCH_SIZE)) > nan_prob, cat_samples, np.nan
)
# @pytest.mark.skip("Skip test that takes long time to run")
def test_nanmixture_inference() -> None:
    nmdo = NanMixtureOutput(GaussianOutput())

    args_proj = nmdo.get_args_proj()
    args_proj.initialize()
    args_proj.hybridize()

    input = mx.nd.ones((BATCH_SIZE))

    trainer = mx.gluon.Trainer(
        args_proj.collect_params(), "sgd", {"learning_rate": 0.00001}
    )

    mixture_samples = mx.nd.array(np_samples)

    N = 1000
    t = tqdm(list(range(N)))
    for _ in t:
        with mx.autograd.record():
            distr_args = args_proj(input)
            d = nmdo.distribution(distr_args)
            loss = d.loss(mixture_samples)
        loss.backward()

        loss_value = loss.mean().asnumpy()
        t.set_postfix({"loss": loss_value})
        trainer.step(BATCH_SIZE)

    distr_args = args_proj(input)
    d = nmdo.distribution(distr_args)

    mu_hat = d.distribution.mu.asnumpy()
    sigma_hat = d.distribution.sigma.asnumpy()
    nan_prob_hat = d.nan_prob.asnumpy()

    assert (
        np.abs(mu - mu_hat) < TOL
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(sigma - sigma_hat) < TOL
    ), f"sigma did not match: sigma = {sigma}, sigma_hat = {sigma_hat}"
    assert (
        np.abs(nan_prob - nan_prob_hat) < TOL
    ), f"nan_prob did not match: nan_prob = {nan_prob}, nan_prob_hat = {nan_prob_hat}"

    nmdo = NanMixtureOutput(CategoricalOutput(3))

    args_proj = nmdo.get_args_proj()
    args_proj.initialize()
    args_proj.hybridize()

    input = mx.nd.ones((BATCH_SIZE))

    trainer = mx.gluon.Trainer(
        args_proj.collect_params(), "sgd", {"learning_rate": 0.000002}
    )

    mixture_samples = mx.nd.array(cat_samples)

    N = 10000
    t = tqdm(list(range(N)))
    for _ in t:
        with mx.autograd.record():
            distr_args = args_proj(input)
            d = nmdo.distribution(distr_args)
            loss = d.loss(mixture_samples)
        loss.backward()
        loss_value = loss.mean().asnumpy()
        t.set_postfix({"loss": loss_value})
        trainer.step(BATCH_SIZE)

    distr_args = args_proj(input)
    d = nmdo.distribution(distr_args)

    cat_probs_hat = d.distribution.probs.asnumpy()
    nan_prob_hat = d.nan_prob.asnumpy()

    print(loss_value)
    print(cat_probs_hat)
    print(nan_prob_hat)
    assert np.allclose(
        cat_probs, cat_probs_hat, atol=TOL
    ), f"categorical dist: cat_probs did not match: cat_probs = {cat_probs}, cat_probs_hat = {cat_probs_hat}"
    assert (
        np.abs(nan_prob - nan_prob_hat) < TOL
    ), f"categorical dist: nan_prob did not match: nan_prob = {nan_prob}, nan_prob_hat = {nan_prob_hat}"
