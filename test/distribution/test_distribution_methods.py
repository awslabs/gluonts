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

test_output = {
    'Gaussian': {
        'mean': mx.nd.array([1000.0, -1000.0]),
        'stddev': mx.nd.array([0.1, 1.0]),
        'variance': mx.nd.array([0.01, 1.0]),
    },
    'Laplace': {
        'mean': mx.nd.array([1000.0, -1000.0]),
        'stddev': mx.nd.array([0.14142136, 1.4142135]),
        'variance': mx.nd.array([0.02, 1.9999999]),
    },
    'StudentT': {
        'mean': mx.nd.array([1000.0, -1000.0]),
        'stddev': mx.nd.array([1.3816986, 3.4641016]),
        'variance': mx.nd.array([1.909091, 12.0]),
    },
    'NegativeBinomial': {
        'mean': mx.nd.array([1000.0, 1.0]),
        'stddev': mx.nd.array([1000.4999, 1.7320508]),
        'variance': mx.nd.array([1.001e+06, 3.000e+00]),
    },
    'Uniform': {
        'mean': mx.nd.array([1500.0, -1000.05]),
        'stddev': mx.nd.array([2.8867514e+02, 2.8860467e-02]),
        'variance': mx.nd.array([8.3333336e+04, 8.3292654e-04]),
    },
    'Binned': {
        'mean': mx.nd.array([-0.985, -0.985]),
        'stddev': mx.nd.array([1.377416, 1.377416]),
        'variance': mx.nd.array([1.8972749, 1.8972749]),
    },
}

# TODO: implement stddev methods for MultivariateGaussian and LowrankMultivariateGaussian
DISTRIBUTIONS = [
    Gaussian,
    Laplace,
    StudentT,
    NegativeBinomial,
    Uniform,
    Binned,
]


@pytest.mark.parametrize("distr_class, params", test_cases)
def test_means(distr_class, params) -> None:
    distr = distr_class(**params)
    means = distr.mean
    distr_name = distr.__class__.__name__
    assert means.shape == test_output[distr_name]['mean'].shape
    # asnumpy()  needed to b/c means is all pointers to values
    assert np.allclose(
        means.asnumpy(), test_output[distr_name]['mean'].asnumpy(), atol=1e-2
    )


@pytest.mark.parametrize("distr_class, params", test_cases)
def test_stdevs(distr_class, params) -> None:
    distr = distr_class(**params)
    stddevs = distr.stddev
    distr_name = distr.__class__.__name__
    assert stddevs.shape == test_output[distr_name]['stddev'].shape
    assert np.allclose(
        stddevs.asnumpy(),
        test_output[distr_name]['stddev'].asnumpy(),
        atol=1e-2,
    )


@pytest.mark.parametrize("distr_class, params", test_cases)
def test_variances(distr_class, params) -> None:
    distr = distr_class(**params)
    variances = distr.variance
    distr_name = distr.__class__.__name__
    assert variances.shape == test_output[distr_name]['variance'].shape
    assert np.allclose(
        variances.asnumpy(),
        test_output[distr_name]['variance'].asnumpy(),
        atol=1e-2,
    )
