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
from flaky import flaky

# First-party imports
from gluonts.distribution import (
    Uniform,
    StudentT,
    NegativeBinomial,
    Laplace,
    Gaussian,
    Gamma,
    Beta,
    MultivariateGaussian,
    Poisson,
    PiecewiseLinear,
    Binned,
    TransformedDistribution,
    Dirichlet,
    DirichletMultinomial,
    Categorical,
)
from gluonts.core.serde import dump_json, load_json, dump_code, load_code

from gluonts.testutil import empirical_cdf


test_cases = [
    (
        Gaussian,
        {
            "mu": mx.nd.array([1000.0, -1000.0]),
            "sigma": mx.nd.array([0.1, 1.0]),
        },
    ),
    (
        Gamma,
        {"alpha": mx.nd.array([2.5, 7.0]), "beta": mx.nd.array([1.5, 2.1])},
    ),
    (
        Beta,
        {"alpha": mx.nd.array([2.5, 7.0]), "beta": mx.nd.array([1.5, 2.1])},
    ),
    (
        Laplace,
        {"mu": mx.nd.array([1000.0, -1000.0]), "b": mx.nd.array([0.1, 1.0])},
    ),
    (
        StudentT,
        {
            "mu": mx.nd.array([1000.0, -1000.0]),
            "sigma": mx.nd.array([1.0, 2.0]),
            "nu": mx.nd.array([4.2, 3.0]),
        },
    ),
    (
        NegativeBinomial,
        {"mu": mx.nd.array([1000.0, 1.0]), "alpha": mx.nd.array([1.0, 2.0])},
    ),
    (
        Uniform,
        {
            "low": mx.nd.array([1000.0, -1000.1]),
            "high": mx.nd.array([2000.0, -1000.0]),
        },
    ),
    (
        Binned,
        {
            "bin_log_probs": mx.nd.array(
                [[0.1, 0.2, 0.1, 0.05, 0.2, 0.1, 0.25]]
            )
            .log()
            .repeat(axis=0, repeats=2),
            "bin_centers": mx.nd.array(
                [[-5, -3, -1.2, -0.5, 0, 0.1, 0.2]]
            ).repeat(axis=0, repeats=2),
        },
    ),
    (
        Binned,
        {
            "bin_log_probs": mx.nd.array(
                [[0.1, 0.2, 0.1, 0.05, 0.2, 0.1, 0.25]]
            )
            .log()
            .repeat(axis=0, repeats=2),
            "bin_centers": mx.nd.array(
                [[-5, -3, -1.2, -0.5, 0, 0.1, 0.2]]
            ).repeat(axis=0, repeats=2),
            "label_smoothing": 0.1,
        },
    ),
    (
        Categorical,
        {
            "log_probs": mx.nd.array([[0.1, 0.2, 0.1, 0.05, 0.2, 0.1, 0.25]])
            .log()
            .repeat(axis=0, repeats=2),
        },
    ),
    (Poisson, {"rate": mx.nd.array([1000.0, 0])}),
]


serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


DISTRIBUTIONS_WITH_CDF = [Gaussian, Uniform, Laplace, Binned]
DISTRIBUTIONS_WITH_QUANTILE_FUNCTION = [Gaussian, Uniform, Laplace, Binned]


@pytest.mark.parametrize("distr_class, params", test_cases)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_sampling(distr_class, params, serialize_fn) -> None:
    distr = distr_class(**params)
    distr = serialize_fn(distr)
    samples = distr.sample()
    assert samples.shape == (2,)
    num_samples = 1_000_000
    samples = distr.sample(num_samples)
    assert samples.shape == (num_samples, 2)

    np_samples = samples.asnumpy()
    # avoid accuracy issues with float32 when calculating std
    # see https://github.com/numpy/numpy/issues/8869
    np_samples = np_samples.astype(np.float64)

    assert np.isfinite(np_samples).all()
    assert np.allclose(
        np_samples.mean(axis=0), distr.mean.asnumpy(), atol=1e-2, rtol=1e-2
    )

    emp_std = np_samples.std(axis=0)
    assert np.allclose(emp_std, distr.stddev.asnumpy(), atol=1e-1, rtol=1e-1)

    if distr_class in DISTRIBUTIONS_WITH_CDF:
        emp_cdf, edges = empirical_cdf(np_samples)
        calc_cdf = distr.cdf(mx.nd.array(edges)).asnumpy()
        assert np.allclose(calc_cdf[1:, :], emp_cdf, atol=1e-2)

    if distr_class in DISTRIBUTIONS_WITH_QUANTILE_FUNCTION:
        levels = np.linspace(1.0e-3, 1.0 - 1.0e-3, 100)
        emp_qfunc = np.percentile(np_samples, levels * 100, axis=0)
        calc_qfunc = distr.quantile(mx.nd.array(levels)).asnumpy()
        assert np.allclose(calc_qfunc, emp_qfunc, rtol=1e-1)


test_cases_multivariate = [
    (
        MultivariateGaussian,
        {
            "mu": mx.nd.array([100.0, -1000.0]),
            "L": mx.nd.array([[6.0, 0.0], [0.5, 20.0]]),
        },
        2,
    ),
    (Dirichlet, {"alpha": mx.nd.array([0.2, 0.4, 0.9])}, 3),
    (
        DirichletMultinomial,
        {"dim": 3, "n_trials": 10, "alpha": mx.nd.array([0.2, 0.4, 0.9])},
        3,
    ),
]


@flaky(min_passes=1, max_runs=3)
@pytest.mark.parametrize("distr, params, dim", test_cases_multivariate)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_multivariate_sampling(distr, params, dim, serialize_fn) -> None:
    distr = distr(**params)
    distr = serialize_fn(distr)
    samples = distr.sample()
    assert samples.shape == (dim,)
    samples = distr.sample(num_samples=1)
    assert samples.shape == (1, dim)
    num_samples = 500_000
    samples = distr.sample(num_samples)
    assert samples.shape == (num_samples, dim)

    np_samples = samples.asnumpy()

    assert np.allclose(
        np_samples.mean(axis=0), distr.mean.asnumpy(), atol=1e-2, rtol=1e-2
    )

    assert np.allclose(
        np.cov(np_samples.transpose()),
        distr.variance.asnumpy(),
        atol=1e-1,
        rtol=1e-1,
    )


test_cases_pwl_sqf = [
    (
        PiecewiseLinear,
        {
            "gamma": mx.nd.array([2]).repeat(axis=0, repeats=2),
            "slopes": mx.nd.array([[3, 1, 3, 0.2, 5, 4]]).repeat(
                axis=0, repeats=2
            ),
            "knot_spacings": mx.nd.array(
                [[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]]
            ).repeat(axis=0, repeats=2),
        },
    )
]


@pytest.mark.parametrize("distr, params", test_cases_pwl_sqf)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_piecewise_linear_sampling(distr, params, serialize_fn):
    distr = distr(**params)
    distr = serialize_fn(distr)
    samples = distr.sample()
    assert samples.shape == (2,)
    num_samples = 100_000
    samples = distr.sample(num_samples)
    assert samples.shape == (num_samples, 2)
