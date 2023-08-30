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

import numpy as np
import pytest

from mxnet import nd

from gluonts.mx.block.regularization import (
    ActivationRegularizationLoss,
    TemporalActivationRegularizationLoss,
)


@pytest.mark.parametrize("alpha", [0, 1, 2, 3, 4, 5, 6, 7, 10, 20])
def test_ActivationRegularizationLoss(alpha: float):
    ar = ActivationRegularizationLoss(alpha, batch_axis=0)
    inputs = [
        nd.arange(1000).reshape(10, 10, 10),
        nd.arange(1000).reshape(10, 10, 10),
        nd.arange(1000).reshape(10, 10, 10),
    ]
    ar_result = ar(*inputs)
    outputs = [
        alpha * nd.mean((array * array), axis=0, exclude=True)
        for array in inputs
    ]
    assert np.isclose(nd.add_n(*outputs).asnumpy(), ar_result.asnumpy()).all()


@pytest.mark.parametrize("beta", [0, 1, 2, 3, 4, 5, 6, 7, 10, 20])
def test_TemporalActivationRegularizationLoss(beta: float):
    tar = TemporalActivationRegularizationLoss(beta, time_axis=1, batch_axis=0)
    inputs = [
        nd.arange(1000).reshape(10, 10, 10),
        nd.arange(1000).reshape(10, 10, 10),
        nd.arange(1000).reshape(10, 10, 10),
    ]
    tar_result = tar(*inputs)
    outputs = [
        beta
        * nd.mean(
            (array[:, 1:, :] - array[:, :-1, :]).__pow__(2),
            axis=0,
            exclude=True,
        )
        for array in inputs
    ]
    assert np.isclose(nd.add_n(*outputs).asnumpy(), tar_result.asnumpy()).all()
