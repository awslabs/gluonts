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
import pytest

from gluonts.core.serde import dump_json, load_json

from gluonts.gluonts_tqdm import tqdm
from gluonts.model.common import NPArrayLike
from gluonts.mx import Tensor
from gluonts.mx.distribution import (
    Gamma,
    GammaOutput,
    Gaussian,
    GaussianOutput,
    GenPareto,
    GenParetoOutput,
    LaplaceOutput,
    MixtureDistribution,
    MixtureDistributionOutput,
    MultivariateGaussianOutput,
    StudentT,
    StudentTOutput,
)
from gluonts.mx.distribution.distribution import Distribution
from gluonts.mx.distribution.distribution_output import DistributionOutput
from gluonts.testutil import empirical_cdf

serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


def plot_samples(s: Tensor, bins: int = 100) -> None:
    from matplotlib import pyplot as plt

    s = s.asnumpy()
    plt.hist(s, bins=bins)
    plt.show()


BINS = np.linspace(-5, 5, 100)


def histogram(samples: NPArrayLike) -> np.ndarray:
    h, _ = np.histogram(samples, bins=BINS, density=True)
    return h


def diff(x: NPArrayLike, y: NPArrayLike) -> np.ndarray:
    return np.mean(np.abs(x - y))


NUM_SAMPLES = 1_000
NUM_SAMPLES_LARGE = 1_000_000


SHAPE = (2, 1, 3)

np.random.seed(35120171)
mx.random.seed(35120171)


@pytest.mark.parametrize(
    "distr1, distr2, p",
    [
        (
            Gaussian(
                mu=mx.nd.zeros(shape=SHAPE),
                sigma=1e-3 + 0.2 * mx.nd.ones(shape=SHAPE),
            ),
            Gaussian(
                mu=mx.nd.ones(shape=SHAPE),
                sigma=1e-3 + 0.1 * mx.nd.ones(shape=SHAPE),
            ),
            0.2 * mx.nd.ones(shape=SHAPE),
        ),
        (
            StudentT(
                mu=mx.nd.ones(shape=SHAPE),
                sigma=1e-1 + mx.nd.zeros(shape=SHAPE),
                nu=mx.nd.zeros(shape=SHAPE) + 2.2,
            ),
            Gaussian(
                mu=-mx.nd.ones(shape=SHAPE),
                sigma=1e-1 + mx.nd.zeros(shape=SHAPE),
            ),
            mx.nd.random_uniform(shape=SHAPE),
        ),
        (
            Gaussian(
                mu=mx.nd.array([0.0]),
                sigma=mx.nd.array([1e-3 + 0.2]),
            ),
            Gaussian(
                mu=mx.nd.array([1.0]),
                sigma=mx.nd.array([1e-3 + 0.1]),
            ),
            mx.nd.array([0.2]),
        ),
        # TODO: add a multivariate case here
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_mixture(
    distr1: Distribution, distr2: Distribution, p: Tensor, serialize_fn
) -> None:
    # sample from component distributions, and select samples
    samples1 = distr1.sample(num_samples=NUM_SAMPLES_LARGE)
    samples2 = distr2.sample(num_samples=NUM_SAMPLES_LARGE)

    # TODO: for multivariate case, test should not sample elements from different components in the event_dim dimension
    rand = mx.nd.random.uniform(shape=(NUM_SAMPLES_LARGE, *p.shape))
    choice = (rand < p.expand_dims(axis=0)).broadcast_like(samples1)
    samples_ref = mx.nd.where(choice, samples1, samples2)

    # construct mixture distribution and sample from it

    mixture_probs = mx.nd.stack(p, 1.0 - p, axis=-1)

    mixture = MixtureDistribution(
        mixture_probs=mixture_probs, components=[distr1, distr2]
    )
    mixture = serialize_fn(mixture)

    samples_mix = mixture.sample(num_samples=NUM_SAMPLES_LARGE)

    # check that shapes are right

    assert (
        samples1.shape
        == samples2.shape
        == samples_mix.shape
        == samples_ref.shape
    )

    # check mean and stddev
    calc_mean = mixture.mean.asnumpy()
    calc_std = mixture.stddev.asnumpy()
    sample_mean = samples_mix.asnumpy().mean(axis=0)
    sample_std = samples_mix.asnumpy().std(axis=0)

    assert np.allclose(calc_mean, sample_mean, atol=1e-1)
    assert np.allclose(calc_std, sample_std, atol=2e-1)

    # check that histograms are close
    assert (
        diff(
            histogram(samples_mix.asnumpy()), histogram(samples_ref.asnumpy())
        )
        < 0.05
    )

    # can only calculated cdf for gaussians currently
    if isinstance(distr1, Gaussian) and isinstance(distr2, Gaussian):
        emp_cdf, edges = empirical_cdf(samples_mix.asnumpy())
        calc_cdf = mixture.cdf(mx.nd.array(edges)).asnumpy()
        assert np.allclose(calc_cdf[1:, :], emp_cdf, atol=1e-2)


@pytest.mark.parametrize(
    "distribution_outputs",
    [
        ((GaussianOutput(), GaussianOutput()),),
        ((GaussianOutput(), StudentTOutput(), LaplaceOutput()),),
        ((MultivariateGaussianOutput(3), MultivariateGaussianOutput(3)),),
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_mixture_output(distribution_outputs, serialize_fn) -> None:
    mdo = MixtureDistributionOutput(*distribution_outputs)

    args_proj = mdo.get_args_proj()
    args_proj.initialize()

    input = mx.nd.ones(shape=(512, 30))

    distr_args = args_proj(input)
    d = mdo.distribution(distr_args)
    d = serialize_fn(d)

    samples = d.sample(num_samples=NUM_SAMPLES)

    sample = d.sample()

    assert samples.shape == (NUM_SAMPLES, *sample.shape)

    log_prob = d.log_prob(sample)

    assert log_prob.shape == d.batch_shape


BATCH_SIZE = 10000

zeros = mx.nd.zeros((BATCH_SIZE, 1))
ones = mx.nd.ones((BATCH_SIZE, 1))

mu1 = 0.0
mu2 = 1.0
sigma1 = 0.2
sigma2 = 0.1

p1 = 0.2
p2 = 1.0 - p1

samples1 = np.random.normal(mu1, scale=sigma1, size=(BATCH_SIZE, 1))
samples2 = np.random.normal(mu2, scale=sigma2, size=(BATCH_SIZE, 1))
np_samples = np.where(
    np.random.uniform(size=(BATCH_SIZE, 1)) > p1, samples2, samples1
)

EXPECTED_HIST = histogram(np_samples)


@pytest.mark.skip("Skip test that takes long time to run")
def test_mixture_inference() -> None:
    mdo = MixtureDistributionOutput([GaussianOutput(), GaussianOutput()])

    args_proj = mdo.get_args_proj()
    args_proj.initialize()
    args_proj.hybridize()

    input = mx.nd.ones((BATCH_SIZE, 1))

    distr_args = args_proj(input)
    d = mdo.distribution(distr_args)

    # plot_samples(d.sample())

    trainer = mx.gluon.Trainer(
        args_proj.collect_params(), "sgd", {"learning_rate": 0.02}
    )

    mixture_samples = mx.nd.array(np_samples)

    N = 1000
    t = tqdm(list(range(N)))
    for i in t:
        with mx.autograd.record():
            distr_args = args_proj(input)
            d = mdo.distribution(distr_args)
            loss = d.loss(mixture_samples)
        loss.backward()
        loss_value = loss.mean().asnumpy()
        t.set_postfix({"loss": loss_value})
        trainer.step(BATCH_SIZE)

    distr_args = args_proj(input)
    d = mdo.distribution(distr_args)

    obtained_hist = histogram(d.sample().asnumpy())

    # uncomment to see histograms
    # pl.plot(obtained_hist)
    # pl.plot(EXPECTED_HIST)
    # pl.show()
    assert diff(obtained_hist, EXPECTED_HIST) < 0.5


def fit_mixture_distribution(
    x: Tensor,
    mdo: MixtureDistributionOutput,
    variate_dimensionality: int = 1,
    epochs: int = 1_000,
):
    args_proj = mdo.get_args_proj()
    args_proj.initialize()
    args_proj.hybridize()

    input = mx.nd.ones((variate_dimensionality, 1))

    trainer = mx.gluon.Trainer(
        args_proj.collect_params(), "sgd", {"learning_rate": 0.02}
    )

    t = tqdm(list(range(epochs)))
    for _ in t:
        with mx.autograd.record():
            distr_args = args_proj(input)
            d = mdo.distribution(distr_args)
            loss = d.loss(x).mean()
        loss.backward()
        loss_value = loss.asnumpy()
        t.set_postfix({"loss": loss_value})
        trainer.step(1)

    distr_args = args_proj(input)
    d = mdo.distribution(distr_args)
    return d


@pytest.mark.parametrize(
    "mixture_distribution, mixture_distribution_output, epochs",
    [
        (
            MixtureDistribution(
                mixture_probs=mx.nd.array([[0.6, 0.4]]),
                components=[
                    Gaussian(mu=mx.nd.array([-1.0]), sigma=mx.nd.array([0.2])),
                    Gamma(alpha=mx.nd.array([2.0]), beta=mx.nd.array([0.5])),
                ],
            ),
            MixtureDistributionOutput([GaussianOutput(), GammaOutput()]),
            2_000,
        ),
        (
            MixtureDistribution(
                mixture_probs=mx.nd.array([[0.7, 0.3]]),
                components=[
                    Gaussian(mu=mx.nd.array([-1.0]), sigma=mx.nd.array([0.2])),
                    GenPareto(xi=mx.nd.array([0.6]), beta=mx.nd.array([1.0])),
                ],
            ),
            MixtureDistributionOutput([GaussianOutput(), GenParetoOutput()]),
            2_000,
        ),
    ],
)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
@pytest.mark.skip("Skip test that takes long time to run")
def test_inference_mixture_different_families(
    mixture_distribution: MixtureDistribution,
    mixture_distribution_output: MixtureDistributionOutput,
    epochs: int,
    serialize_fn,
) -> None:
    # First sample from mixture distribution and then confirm the MLE are close to true parameters
    num_samples = 10_000
    samples = mixture_distribution.sample(num_samples=num_samples)
    variate_dimensionality = (
        mixture_distribution.components[0].args[0].shape[0]
    )
    fitted_dist = fit_mixture_distribution(
        samples,
        mixture_distribution_output,
        variate_dimensionality,
        epochs=epochs,
    )

    assert np.allclose(
        fitted_dist.mixture_probs.asnumpy(),
        mixture_distribution.mixture_probs.asnumpy(),
        atol=1e-1,
    ), f"Mixing probability estimates {fitted_dist.mixture_probs.asnumpy()} too far from {mixture_distribution.mixture_probs.asnumpy()}"
    for ci, c in enumerate(mixture_distribution.components):
        for ai, a in enumerate(c.args):
            assert np.allclose(
                fitted_dist.components[ci].args[ai].asnumpy(),
                a.asnumpy(),
                atol=1e-1,
            ), f"Parameter {ai} estimate {fitted_dist.components[ci].args[ai].asnumpy()} too far from {c}"


@pytest.mark.parametrize(
    "distribution, values_outside_support, distribution_output",
    [
        (
            Gamma(alpha=mx.nd.array([0.9]), beta=mx.nd.array([2.0])),
            mx.nd.array([-1.0]),
            GammaOutput(),
        ),
        (
            Gamma(alpha=mx.nd.array([0.9]), beta=mx.nd.array([2.0])),
            mx.nd.array([0.0]),
            GammaOutput(),
        ),
        (
            GenPareto(xi=mx.nd.array([1 / 3.0]), beta=mx.nd.array([1.0])),
            mx.nd.array([-1.0]),
            GenParetoOutput(),
        ),
    ],
)
def test_mixture_logprob(
    distribution: Distribution,
    values_outside_support: Tensor,
    distribution_output: DistributionOutput,
) -> None:

    assert np.all(
        ~np.isnan(distribution.log_prob(values_outside_support).asnumpy())
    ), f"{distribution} should return -inf log_probs instead of NaNs"

    p = 0.5
    gaussian = Gaussian(mu=mx.nd.array([0]), sigma=mx.nd.array([2.0]))
    mixture = MixtureDistribution(
        mixture_probs=mx.nd.array([[p, 1 - p]]),
        components=[gaussian, distribution],
    )
    lp = mixture.log_prob(values_outside_support)
    assert np.allclose(
        lp.asnumpy(),
        np.log(p) + gaussian.log_prob(values_outside_support).asnumpy(),
        atol=1e-6,
    ), f"log_prob(x) should be equal to log(p)+gaussian.log_prob(x)"

    fit_mixture = fit_mixture_distribution(
        values_outside_support,
        MixtureDistributionOutput([GaussianOutput(), distribution_output]),
        variate_dimensionality=1,
        epochs=3,
    )
    for ci, c in enumerate(fit_mixture.components):
        for ai, a in enumerate(c.args):
            assert ~np.isnan(a.asnumpy()), f"NaN gradients led to {c}"
