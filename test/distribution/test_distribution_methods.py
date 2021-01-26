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

from gluonts.mx.distribution import (
    Beta,
    Binned,
    Categorical,
    Gamma,
    Gaussian,
    GenPareto,
    Laplace,
    MultivariateGaussian,
    NegativeBinomial,
    OneInflatedBeta,
    PiecewiseLinear,
    Poisson,
    StudentT,
    TransformedDistribution,
    Uniform,
    ZeroAndOneInflatedBeta,
    ZeroInflatedBeta,
    ZeroInflatedPoissonOutput,
)

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
        GenPareto,
        {
            "xi": mx.nd.array([1 / 3.0, 1 / 4.0]),
            "beta": mx.nd.array([1.0, 2.0]),
        },
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
                [[1e-300, 0.3, 0.1, 0.05, 0.2, 0.1, 0.25]]
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
                [[1e-300, 0.3, 0.1, 0.05, 0.2, 0.1, 0.25]]
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
            "log_probs": mx.nd.array(
                [[1e-300, 0.3, 0.1, 0.05, 0.2, 0.1, 0.25]]
            )
            .log()
            .repeat(axis=0, repeats=2),
        },
    ),
    (Poisson, {"rate": mx.nd.array([1000.0, 1.0])}),
    (
        ZeroInflatedBeta,
        {
            "alpha": mx.nd.array([0.175]),
            "beta": mx.nd.array([0.6]),
            "zero_probability": mx.nd.array([0.3]),
        },
    ),
    (
        OneInflatedBeta,
        {
            "alpha": mx.nd.array([0.175]),
            "beta": mx.nd.array([0.6]),
            "one_probability": mx.nd.array([0.3]),
        },
    ),
    (
        ZeroAndOneInflatedBeta,
        {
            "alpha": mx.nd.array([0.175]),
            "beta": mx.nd.array([0.6]),
            "zero_probability": mx.nd.array([0.3]),
            "one_probability": mx.nd.array([0.3]),
        },
    ),
]

test_output = {
    "Gaussian": {
        "mean": mx.nd.array([1000.0, -1000.0]),
        "stddev": mx.nd.array([0.1, 1.0]),
        "variance": mx.nd.array([0.01, 1.0]),
    },
    "Beta": {
        "mean": mx.nd.array([0.625, 0.7692307]),
        "stddev": mx.nd.array([0.2165063, 0.1325734]),
        "variance": mx.nd.array([0.046875, 0.0175757]),
    },
    "Gamma": {
        "mean": mx.nd.array([1.6666666, 3.3333333]),
        "stddev": mx.nd.array([1.05409255, 1.25988158]),
        "variance": mx.nd.array([1.1111111, 1.58730159]),
    },
    "GenPareto": {
        "mean": mx.nd.array([1.5, 2.666666666666666]),
        "stddev": mx.nd.array([2.5980762, 3.7712361663282534]),
        "variance": mx.nd.array([6.75, 14.222222222222221]),
    },
    "Laplace": {
        "mean": mx.nd.array([1000.0, -1000.0]),
        "stddev": mx.nd.array([0.14142136, 1.4142135]),
        "variance": mx.nd.array([0.02, 1.9999999]),
    },
    "StudentT": {
        "mean": mx.nd.array([1000.0, -1000.0]),
        "stddev": mx.nd.array([1.3816986, 3.4641016]),
        "variance": mx.nd.array([1.909091, 12.0]),
    },
    "NegativeBinomial": {
        "mean": mx.nd.array([1000.0, 1.0]),
        "stddev": mx.nd.array([1000.4999, 1.7320508]),
        "variance": mx.nd.array([1.001e06, 3.000e00]),
    },
    "Uniform": {
        "mean": mx.nd.array([1500.0, -1000.05]),
        "stddev": mx.nd.array([2.8867514e02, 2.8860467e-02]),
        "variance": mx.nd.array([8.3333336e04, 8.3292654e-04]),
    },
    "Binned": {
        "mean": mx.nd.array([-0.985, -0.985]),
        "stddev": mx.nd.array([1.377416, 1.377416]),
        "variance": mx.nd.array([1.8972749, 1.8972749]),
    },
    "Categorical": {
        "mean": mx.nd.array([3.45, 3.45]),
        "stddev": mx.nd.array([1.9868319, 1.9868319]),
        "variance": mx.nd.array([3.947501, 3.947501]),
    },
    "Poisson": {
        "mean": mx.nd.array([1000.0, 1.0]),
        "stddev": mx.nd.array([31.622776, 1.0]),
        "variance": mx.nd.array([1000.0, 1.0]),
    },
    "ZeroInflatedBeta": {
        "mean": mx.nd.array([0.15806451612903227]),
        "stddev": mx.nd.array([0.2822230782496945]),
        "variance": mx.nd.array([0.07964986589673317]),
    },
    "OneInflatedBeta": {
        "mean": mx.nd.array([0.45806451612903226]),
        "stddev": mx.nd.array([0.44137416804715]),
        "variance": mx.nd.array([0.19481115621931383]),
    },
    "ZeroAndOneInflatedBeta": {
        "mean": mx.nd.array([0.3903225806451613]),
        "stddev": mx.nd.array([0.45545503304667967]),
        "variance": mx.nd.array([0.20743928712755205]),
    },
}

test_cases_quantile = [
    (
        Gaussian,
        {
            "mu": mx.nd.array([0.0]),
            "sigma": mx.nd.array([1.0]),
        },
    ),
    (
        GenPareto,
        {
            "xi": mx.nd.array([1 / 3.0]),
            "beta": mx.nd.array([1.0]),
        },
    ),
]

test_output_quantile = {
    "Gaussian": {
        "x": mx.nd.array([3.0902362]),
        "cdf": mx.nd.array([0.999]),
        "level": mx.nd.array([0.999]),
        "quantile": mx.nd.array([[3.0902362]]),
    },
    "GenPareto": {
        "x": mx.nd.array([26.99999999999998]),
        "cdf": mx.nd.array([0.999]),
        "level": mx.nd.array([0.999]),
        "quantile": mx.nd.array([[26.99999999999998]]),
    },
}

# TODO: implement stddev methods for MultivariateGaussian and LowrankMultivariateGaussian
DISTRIBUTIONS = [
    Gaussian,
    Laplace,
    StudentT,
    Gamma,
    NegativeBinomial,
    Uniform,
    Binned,
    Poisson,
    ZeroInflatedBeta,
    OneInflatedBeta,
    ZeroAndOneInflatedBeta,
]


serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


@pytest.mark.parametrize("distr_class, params", test_cases)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_means(distr_class, params, serialize_fn) -> None:
    distr = distr_class(**params)
    distr = serialize_fn(distr)
    means = distr.mean
    distr_name = distr.__class__.__name__
    assert means.shape == test_output[distr_name]["mean"].shape
    # asnumpy()  needed to b/c means is all pointers to values
    assert np.allclose(
        means.asnumpy(), test_output[distr_name]["mean"].asnumpy(), atol=1e-11
    )


@pytest.mark.parametrize("distr_class, params", test_cases)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_stdevs(distr_class, params, serialize_fn) -> None:
    distr = distr_class(**params)
    distr = serialize_fn(distr)
    stddevs = distr.stddev
    distr_name = distr.__class__.__name__
    assert stddevs.shape == test_output[distr_name]["stddev"].shape
    assert np.allclose(
        stddevs.asnumpy(),
        test_output[distr_name]["stddev"].asnumpy(),
        atol=1e-11,
    )


@pytest.mark.parametrize("distr_class, params", test_cases)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_variances(distr_class, params, serialize_fn) -> None:
    distr = distr_class(**params)
    distr = serialize_fn(distr)
    variances = distr.variance
    distr_name = distr.__class__.__name__
    assert variances.shape == test_output[distr_name]["variance"].shape
    assert np.allclose(
        variances.asnumpy(),
        test_output[distr_name]["variance"].asnumpy(),
        atol=1e-11,
    )


@pytest.mark.parametrize("distr_class, params", test_cases_quantile)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_quantile(distr_class, params, serialize_fn) -> None:
    distr = distr_class(**params)
    distr = serialize_fn(distr)
    distr_name = distr.__class__.__name__
    quantile = distr.quantile(test_output_quantile[distr_name]["level"])
    assert np.allclose(
        quantile.asnumpy(),
        test_output_quantile[distr_name]["quantile"].asnumpy(),
        atol=1e-11,
    )


@pytest.mark.parametrize("distr_class, params", test_cases_quantile)
@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_cdf(distr_class, params, serialize_fn) -> None:
    distr = distr_class(**params)
    distr = serialize_fn(distr)
    distr_name = distr.__class__.__name__
    cdf = distr.cdf(test_output_quantile[distr_name]["x"])
    assert np.allclose(
        cdf.asnumpy(),
        test_output_quantile[distr_name]["cdf"].asnumpy(),
        atol=1e-11,
    )
