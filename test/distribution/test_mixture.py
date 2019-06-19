# Third-party imports
import mxnet as mx
import numpy as np
import pytest

# First-party imports
from gluonts.gluonts_tqdm import tqdm
from gluonts.model.common import Tensor, NPArrayLike
from gluonts.distribution.distribution import Distribution
from gluonts.distribution import (
    Gaussian,
    StudentT,
    MixtureDistribution,
    GaussianOutput,
    StudentTOutput,
    LaplaceOutput,
    MultivariateGaussianOutput,
    MixtureDistributionOutput,
)
from gluonts.testutil import empirical_cdf


def plot_samples(s: Tensor, bins: int = 100) -> None:
    from matplotlib import pyplot as plt

    s = s.asnumpy()
    plt.hist(s, bins=bins)
    plt.show()


BINS = np.linspace(-5, 5, 100)


def histogram(samples: NPArrayLike) -> np.ndarray:
    h, _ = np.histogram(samples, bins=BINS, normed=True)
    return h


def diff(x: NPArrayLike, y: NPArrayLike) -> np.ndarray:
    return np.mean(np.abs(x - y))


NUM_SAMPLES = 1_000
NUM_SAMPLES_LARGE = 100_000


SHAPE = (2, 1, 3)


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
        # TODO: add a multivariate case here
    ],
)
def test_mixture(
    distr1: Distribution, distr2: Distribution, p: Tensor
) -> None:

    # sample from component distributions, and select samples

    samples1 = distr1.sample(num_samples=NUM_SAMPLES_LARGE)
    samples2 = distr2.sample(num_samples=NUM_SAMPLES_LARGE)

    rand = mx.nd.random.uniform(shape=(NUM_SAMPLES_LARGE, *p.shape))
    choice = (rand < p.expand_dims(axis=0)).broadcast_like(samples1)
    samples_ref = mx.nd.where(choice, samples1, samples2)

    # construct mixture distribution and sample from it

    mixture_probs = mx.nd.stack(p, 1.0 - p, axis=-1)

    mixture = MixtureDistribution(
        mixture_probs=mixture_probs, components=[distr1, distr2]
    )

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
    sample_mean = samples_mix.asnumpy().mean(axis=0)

    assert np.allclose(calc_mean, sample_mean, atol=1e-1)

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
def test_mixture_output(distribution_outputs) -> None:
    mdo = MixtureDistributionOutput(*distribution_outputs)

    args_proj = mdo.get_args_proj()
    args_proj.initialize()

    input = mx.nd.ones(shape=(512, 30))

    distr_args = args_proj(input)
    d = mdo.distribution(distr_args)

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


@pytest.mark.timeout(20)
@pytest.mark.skip('Skip test that takes long time to run')
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
        args_proj.collect_params(), 'sgd', {'learning_rate': 0.02}
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
        t.set_postfix({'loss': loss_value})
        trainer.step(BATCH_SIZE)

    distr_args = args_proj(input)
    d = mdo.distribution(distr_args)

    obtained_hist = histogram(d.sample().asnumpy())

    # uncomment to see histograms
    # pl.plot(obtained_hist)
    # pl.plot(EXPECTED_HIST)
    # pl.show()
    assert diff(obtained_hist, EXPECTED_HIST) < 0.5
