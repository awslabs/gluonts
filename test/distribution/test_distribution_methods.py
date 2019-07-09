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
}

# TODO: work out all means, stddevs, variances as similar to test_cases structure
DISTRIBUTIONS_WITH_MEAN = [Gaussian, Laplace, StudentT, NegativeBinomial]


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
