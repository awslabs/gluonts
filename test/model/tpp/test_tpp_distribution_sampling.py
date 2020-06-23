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
from gluonts.model.tpp.distribution import Loglogistic, Weibull
from gluonts.core.serde import dump_json, load_json

from gluonts.testutil import empirical_cdf


test_cases = [
    (
        Loglogistic,
        {"mu": mx.nd.array([-1.0, 0.75]), "sigma": mx.nd.array([0.1, 0.3])},
    ),
    (
        Weibull,
        {"rate": mx.nd.array([0.5, 2.0]), "shape": mx.nd.array([1.5, 5.0])},
    ),
]


serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


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

    # Check the survival function
    emp_cdf, edges = empirical_cdf(np_samples)
    calc_cdf = 1.0 - distr.log_survival(mx.nd.array(edges)).exp().asnumpy()
    assert np.allclose(calc_cdf[1:, :], emp_cdf, atol=1e-2)
