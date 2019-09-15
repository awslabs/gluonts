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
import mxnet.ndarray as nd
import numpy as np
import pytest

# First-party imports
from gluonts.distribution import Uniform
from gluonts.distribution.transformed_distribution import (
    TransformedDistribution,
)
from gluonts.distribution import bijection
from gluonts.core.serde import dump_json, load_json


serialize_fn_list = [lambda x: x, lambda x: load_json(dump_json(x))]


def exp_cdf(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-x)


@pytest.mark.parametrize("serialize_fn", serialize_fn_list)
def test_transformed_distribution(serialize_fn) -> None:
    zero = nd.zeros(1)
    one = nd.ones(1)

    # If Y = -log(U) with U ~ Uniform(0, 1), then Y ~ Exponential(1)
    exponential = TransformedDistribution(
        Uniform(zero, one),
        [bijection.log, bijection.AffineTransformation(scale=-1 * one)],
    )
    exponential = serialize_fn(exponential)

    # For Y ~ Exponential(1), P(Y) = e^{-x) ==> log P(Y) = -x
    assert exponential.log_prob(1 * one).asscalar() == -1.0
    assert exponential.log_prob(2 * one).asscalar() == -2.0

    v = np.linspace(0, 5, 101)
    assert np.allclose(exponential.cdf(nd.array(v)).asnumpy(), exp_cdf(v))

    # If Y ~ Exponential(1), then U = 1 - e^{-Y} has Uniform(0, 1) distribution
    uniform = TransformedDistribution(
        exponential,
        [
            bijection.AffineTransformation(scale=-1 * one),
            bijection.log.inverse_bijection(),  # == bijection.exp
            bijection.AffineTransformation(loc=one, scale=-1 * one),
        ],
    )
    uniform = serialize_fn(uniform)
    # For U ~ Uniform(0, 1), log P(U) = 0
    assert uniform.log_prob(0.5 * one).asscalar() == 0
    assert uniform.log_prob(0.2 * one).asscalar() == 0

    v = np.linspace(0, 1, 101)
    assert np.allclose(uniform.cdf(nd.array(v)).asnumpy(), v)
