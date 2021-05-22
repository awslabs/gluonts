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

from typing import Dict

import mxnet as mx
import numpy as np
import pytest

from gluonts.core.serde import dump_json, load_json

from gluonts.gluonts_tqdm import tqdm
from gluonts.model.common import NPArrayLike
from gluonts.mx import Tensor
from gluonts.mx.distribution import (
    Categorical,
    CategoricalOutput,
    Gaussian,
    GaussianOutput,
    NanMixture,
    NanMixtureOutput,
    StudentTOutput,
)
from gluonts.mx.distribution.distribution import Distribution

serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


def diff(x: NPArrayLike, y: NPArrayLike) -> np.ndarray:
    return np.mean(np.abs(x - y))


NUM_SAMPLES = 2_000
NUM_SAMPLES_LARGE = 100_000


# prepare test cases and calculate desired gradients analytically
p = np.array([[[0.0001, 0.0001, 0.5]], [[0.999, 0.999, 0.5]]])
SHAPE = p.shape

x_gauss = np.array([[[np.nan, 3.5, -0.5]], [[np.nan, 3.5, np.nan]]])
mu = np.zeros(SHAPE)
sigma = 1e-3 + np.ones(shape=SHAPE)
params_gauss = {"mu": mx.nd.array(mu), "sigma": mx.nd.array(sigma)}

mu_grad_true = -(x_gauss - mu) / np.square(sigma)
mu_grad_true[x_gauss != x_gauss] = 0
mu_grad_true[p == 1] = 0

sigma_grad_true = -(
    np.square(mu) - 2 * mu * x_gauss - np.square(sigma) + np.square(x_gauss)
) / (sigma ** 3)

sigma_grad_true[x_gauss != x_gauss] = 0
sigma_grad_true[p == 1] = 0
params_gauss_grad = {"mu": mu_grad_true, "sigma": sigma_grad_true}

p_cat = np.array(
    [
        [[[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]]],
        [[[0.9, 0.1], [0.05, 0.95], [0.45, 0.55]]],
    ]
)
params_cat = {"log_probs": mx.nd.array(np.log(p_cat))}

x_cat = np.array([[[np.nan, 0, 1]], [[np.nan, 0, np.nan]]])
p_cat_zero = p_cat[:, :, :, 0]

log_p_cat_grad_zero_true = np.where(
    x_cat == 0,
    -np.ones(p_cat_zero.shape, dtype=np.float),
    np.zeros(p_cat_zero.shape, dtype=np.float),
)
p_cat_one = p_cat[:, :, :, 1]
log_p_cat_grad_one_true = np.where(
    x_cat == 1,
    -np.ones(p_cat_one.shape, dtype=np.float),
    np.zeros(p_cat_one.shape, dtype=np.float),
)
log_p_cat_grad_true = np.stack(
    [log_p_cat_grad_zero_true, log_p_cat_grad_one_true], axis=-1
)

params_cat_grad = {"log_probs": log_p_cat_grad_true}


@pytest.mark.parametrize(
    "distr_class, p, x, distr_params, distr_params_grad",
    [
        (
            Gaussian,
            mx.nd.array(p),
            mx.nd.array(x_gauss),
            params_gauss,
            params_gauss_grad,
        ),
        (
            Categorical,
            mx.nd.array(p),
            mx.nd.array(x_cat),
            params_cat,
            params_cat_grad,
        ),
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_nan_mixture(
    distr_class,
    p: Tensor,
    x: Tensor,
    distr_params: Dict[str, Tensor],
    distr_params_grad: Dict[str, Tensor],
    serialize_fn,
) -> None:
    # sample from component distributions, and select samples
    distr = distr_class(**distr_params)

    samples = distr.sample(num_samples=NUM_SAMPLES_LARGE)

    rand = mx.nd.random.uniform(shape=(NUM_SAMPLES_LARGE, *p.shape))
    choice = (rand > p.expand_dims(axis=0)).broadcast_like(samples)
    samples_ref = mx.nd.where(choice, samples, samples.zeros_like())

    # construct NanMixture distribution and sample from it
    nan_mixture = NanMixture(nan_prob=p, distribution=distr)

    nan_mixture = serialize_fn(nan_mixture)

    samples_mix = nan_mixture.sample(num_samples=NUM_SAMPLES_LARGE)
    # check that shapes are right

    assert samples.shape == samples_mix.shape == samples_ref.shape

    # TODO check mean and stddev

    # check log_prob
    log_prob = nan_mixture.log_prob(x)

    log_prob_true = mx.nd.log(mx.nd.where(x != x, p, (1 - p) * distr.prob(x)))

    assert np.allclose(log_prob.asnumpy(), log_prob_true.asnumpy())

    for param in distr_params:
        distr_params[param].attach_grad()
    p.attach_grad()

    with mx.autograd.record():
        distr = distr_class(**distr_params)
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

    assert np.allclose(p.grad.asnumpy(), p_grad_true.asnumpy())

    for param in distr_params:

        assert np.allclose(
            distr_params[param].grad.asnumpy(), distr_params_grad[param]
        )


NUM_SAMPLES = 10000
mu = 1.0
sigma = 1.5
TOL = 0.01

nan_prob = 0.3

np.random.seed(1)

samples = np.random.normal(mu, scale=sigma, size=(NUM_SAMPLES))
np_samples = np.where(
    np.random.uniform(size=(NUM_SAMPLES)) > nan_prob, samples, np.nan
)


@pytest.mark.skip("Skip test that takes long time to run")
def test_nanmixture_gaussian_inference() -> None:

    nmdo = NanMixtureOutput(GaussianOutput())

    args_proj = nmdo.get_args_proj()
    args_proj.initialize()
    args_proj.hybridize()

    input = mx.nd.ones((NUM_SAMPLES))

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
        trainer.step(NUM_SAMPLES)

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


n_cat = 3
cat_probs = np.array([0.2, 0.3, 0.5])
cat_samples = np.random.choice(
    list(range(n_cat)), p=cat_probs, size=NUM_SAMPLES
)
cat_samples = np.where(
    np.random.uniform(size=(NUM_SAMPLES)) > nan_prob, cat_samples, np.nan
)


@pytest.mark.skip("Skip test that takes long time to run")
def test_nanmixture_categorical_inference() -> None:

    nmdo = NanMixtureOutput(CategoricalOutput(3))

    args_proj = nmdo.get_args_proj()
    args_proj.initialize()
    args_proj.hybridize()

    input = mx.nd.ones((NUM_SAMPLES))

    trainer = mx.gluon.Trainer(
        args_proj.collect_params(), "sgd", {"learning_rate": 0.000002}
    )

    mixture_samples = mx.nd.array(cat_samples)

    N = 3000
    t = tqdm(list(range(N)))
    for _ in t:
        with mx.autograd.record():
            distr_args = args_proj(input)
            d = nmdo.distribution(distr_args)
            loss = d.loss(mixture_samples)
        loss.backward()
        loss_value = loss.mean().asnumpy()
        t.set_postfix({"loss": loss_value})
        trainer.step(NUM_SAMPLES)

    distr_args = args_proj(input)
    d = nmdo.distribution(distr_args)

    cat_probs_hat = d.distribution.probs.asnumpy()
    nan_prob_hat = d.nan_prob.asnumpy()

    assert np.allclose(
        cat_probs, cat_probs_hat, atol=TOL
    ), f"categorical dist: cat_probs did not match: cat_probs = {cat_probs}, cat_probs_hat = {cat_probs_hat}"
    assert (
        np.abs(nan_prob - nan_prob_hat) < TOL
    ), f"categorical dist: nan_prob did not match: nan_prob = {nan_prob}, nan_prob_hat = {nan_prob_hat}"


n_cat = 3
cat_probs = np.array([0.2, 0.3, 0.5])
cat_samples = np.random.choice(
    list(range(n_cat)), p=cat_probs, size=NUM_SAMPLES
)
cat_samples = np.where(
    np.random.uniform(size=(NUM_SAMPLES)) > nan_prob, cat_samples, np.nan
)


@pytest.mark.parametrize(
    "distribution_output",
    [
        GaussianOutput(),
        StudentTOutput(),
        CategoricalOutput(num_cats=2),
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_nanmixture_output(distribution_output, serialize_fn) -> None:

    nmdo = NanMixtureOutput(distribution_output)

    args_proj = nmdo.get_args_proj()
    args_proj.initialize()

    input = mx.nd.ones(shape=(3, 2))

    distr_args = args_proj(input)

    d = nmdo.distribution(distr_args)
    d = serialize_fn(d)

    samples = d.sample(num_samples=NUM_SAMPLES)

    sample = d.sample()

    assert samples.shape == (NUM_SAMPLES, *sample.shape)

    log_prob = d.log_prob(sample)

    assert log_prob.shape == d.batch_shape
