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

from typing import List, Tuple, Union

# Third-party imports
import pytest
from mxnet import nd

# First-party imports
from gluonts.model.common import Tensor
from gluonts.model.tpp.distribution import (
    TPPDistributionOutput,
    LoglogisticOutput,
    WeibullOutput,
)


@pytest.mark.parametrize(
    "distr_out, data, scale, expected_batch_shape, expected_event_shape",
    [
        (
            LoglogisticOutput(),
            nd.random.normal(shape=(3, 4, 5, 6)),
            [None, nd.ones(shape=(1,)), nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
        (
            WeibullOutput(),
            nd.random.gamma(shape=(3, 4, 5, 6)),
            # [None, nd.array([5.0])],
            [None, nd.ones(shape=(1,)), nd.ones(shape=(3, 4, 5))],
            (3, 4, 5),
            (),
        ),
    ],
)
def test_distribution_output_shapes(
    distr_out: TPPDistributionOutput,
    data: Tensor,
    scale: List[Union[None, Tensor]],
    expected_batch_shape: Tuple,
    expected_event_shape: Tuple,
):
    args_proj = distr_out.get_args_proj()
    args_proj.initialize()

    args = args_proj(data)

    assert distr_out.event_shape == expected_event_shape

    for s in scale:
        distr = distr_out.distribution(args, scale=s)

        assert distr.batch_shape == expected_batch_shape
        assert distr.event_shape == expected_event_shape

        x = distr.sample()

        assert x.shape == distr.batch_shape + distr.event_shape

        loss = distr.loss(x)

        assert loss.shape == distr.batch_shape

        x1 = distr.sample(num_samples=1)

        assert x1.shape == (1,) + distr.batch_shape + distr.event_shape

        x3 = distr.sample(num_samples=3)

        assert x3.shape == (3,) + distr.batch_shape + distr.event_shape
