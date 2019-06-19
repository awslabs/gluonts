# Third-party imports
import mxnet as mx
import numpy as np
import pytest

# First-party imports
from gluonts.distribution import (
    Uniform,
    StudentT,
    NegativeBinomial,
    Laplace,
    Gaussian,
    MultivariateGaussian,
    PiecewiseLinear,
    Binned,
    TransformedDistribution,
)

from gluonts.testutil import empirical_cdf


test_cases = [
    (
        Gaussian,
        {
            'mu': mx.nd.array([1000.0, -1000.0]),
            'sigma': mx.nd.array([0.1, 1.0]),
        },
    ),
    (
        Laplace,
        {'mu': mx.nd.array([1000.0, -1000.0]), 'b': mx.nd.array([0.1, 1.0])},
    ),
    (
        StudentT,
        {
            'mu': mx.nd.array([1000.0, -1000.0]),
            'sigma': mx.nd.array([1.0, 2.0]),
            'nu': mx.nd.array([4.2, 3.0]),
        },
    ),
    (
        NegativeBinomial,
        {'mu': mx.nd.array([1000.0, 1.0]), 'alpha': mx.nd.array([1.0, 2.0])},
    ),
    (
        Uniform,
        {
            'low': mx.nd.array([1000.0, -1000.1]),
            'high': mx.nd.array([2000.0, -1000.0]),
        },
    ),
    (
        Binned,
        {
            'bin_probs': mx.nd.array(
                [[0, 0.3, 0.1, 0.05, 0.2, 0.1, 0.25]]
            ).repeat(axis=0, repeats=2),
            'bin_centers': mx.nd.array(
                [[-5, -3, -1.2, -0.5, 0, 0.1, 0.2]]
            ).repeat(axis=0, repeats=2),
        },
    ),
]


DISTRIBUTIONS_WITH_CDF = [Gaussian, Uniform, Laplace, Binned]


@pytest.mark.parametrize("distr_class, params", test_cases)
def test_sampling(distr_class, params) -> None:
    distr = distr_class(**params)
    samples = distr.sample()
    assert samples.shape == (2,)
    num_samples = 100_000
    samples = distr.sample(num_samples)
    assert samples.shape == (num_samples, 2)

    np_samples = samples.asnumpy()

    assert np.isfinite(np_samples).all()

    assert np.allclose(
        np_samples.mean(axis=0), distr.mean.asnumpy(), atol=1e-2, rtol=1e-2
    )

    emp_std = np_samples.std(axis=0)
    assert np.allclose(emp_std, distr.stddev.asnumpy(), atol=1e-1, rtol=5e-2)

    if distr_class in DISTRIBUTIONS_WITH_CDF:
        emp_cdf, edges = empirical_cdf(np_samples)
        calc_cdf = distr.cdf(mx.nd.array(edges)).asnumpy()
        assert np.allclose(calc_cdf[1:, :], emp_cdf, atol=1e-2)


test_cases_multivariate = [
    (
        MultivariateGaussian,
        {
            'mu': mx.nd.array([100.0, -1000.0]),
            'L': mx.nd.array([[6.0, 0.0], [0.5, 20.0]]),
        },
    )
]


@pytest.mark.parametrize("distr, params", test_cases_multivariate)
def test_multivariate_sampling(distr, params) -> None:
    distr = distr(**params)
    samples = distr.sample()
    assert samples.shape == (2,)
    num_samples = 100_000
    samples = distr.sample(num_samples)
    assert samples.shape == (num_samples, 2)

    np_samples = samples.asnumpy()

    assert np.allclose(
        np_samples.mean(axis=0), params['mu'].asnumpy(), atol=1e-2, rtol=1e-2
    )
    assert np.allclose(
        np.linalg.cholesky(np.cov(np_samples.transpose())),
        params['L'].asnumpy(),
        atol=1e-1,
        rtol=1e-1,
    )
    # TODO: should we move this to a different test?
    # TODO: we probably need unit tests for .mean, .variance anyway
    assert np.allclose(
        np.linalg.cholesky(distr.variance.asnumpy()), params['L'].asnumpy()
    )


test_cases_pwl_sqf = [
    (
        PiecewiseLinear,
        {
            'gamma': mx.nd.array([2]).repeat(axis=0, repeats=2),
            'slopes': mx.nd.array([[3, 1, 3, 0.2, 5, 4]]).repeat(
                axis=0, repeats=2
            ),
            'knot_spacings': mx.nd.array(
                [[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]]
            ).repeat(axis=0, repeats=2),
        },
    )
]


@pytest.mark.parametrize("distr, params", test_cases_pwl_sqf)
def test_piecewise_linear_sampling(distr, params):
    distr = distr(**params)
    samples = distr.sample()
    assert samples.shape == (2,)
    num_samples = 100_000
    samples = distr.sample(num_samples)
    assert samples.shape == (num_samples, 2)
