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

from gluonts.mx.trainer.model_averaging import (
    SelectNBestMean,
    SelectNBestSoftmax,
)


@pytest.mark.parametrize("strategy", [SelectNBestMean, SelectNBestSoftmax])
@pytest.mark.parametrize("num_models", [1, 2])
def test_model_averaging(strategy, num_models):
    total_models = 2

    # model 1
    param_1 = {
        "arg1": mx.nd.array([[1, 2, 3], [1, 2, 3]]),
        "arg2": mx.nd.array([[1, 2], [1, 2]]),
    }
    loss_1 = 1

    # model 2
    param_2 = {
        "arg1": mx.nd.array([[1, 1, 1], [1, 1, 1]]),
        "arg2": mx.nd.array([[1, 1], [1, 1]]),
    }
    loss_2 = 1.5
    assert (
        loss_1 < loss_2
    )  # to keep it simple we assume that the first model has lower loss than the second

    # combine models
    all_arg_params = [param_1, param_2]
    dummy_checkpoints = [
        {
            "params_path": "dummy_path",
            "epoch_no": 0,
            "score": loss_1,
        },
        {
            "params_path": "dummy_path",
            "epoch_no": 0,
            "score": loss_2,
        },
    ]

    # compute expected weights
    avg = strategy(num_models=num_models)
    _, weights = avg.select_checkpoints(dummy_checkpoints)
    assert len(weights) == num_models

    if isinstance(avg, SelectNBestMean):
        exp_weights = [1 / num_models for _ in range(num_models)]
        assert weights == exp_weights
    elif isinstance(avg, SelectNBestSoftmax):
        losses = [c["score"] for c in dummy_checkpoints]
        losses = sorted(losses)[:num_models]
        exp_weights = [np.exp(-l) for l in losses]
        exp_weights = [x / sum(exp_weights) for x in exp_weights]
        assert weights == exp_weights

    # compute expected output
    weights = weights + [0] * (
        total_models - num_models
    )  # include 0 weights for the models that are not averaged
    exp_output = {
        "arg1": weights[0] * mx.nd.array([[1, 2, 3], [1, 2, 3]])
        + weights[1] * mx.nd.array([[1, 1, 1], [1, 1, 1]]),
        "arg2": weights[0] * mx.nd.array([[1, 2], [1, 2]])
        + weights[1] * mx.nd.array([[1, 1], [1, 1]]),
    }

    avg_params = {}
    for k in all_arg_params[0]:
        arrays = [p[k] for p in all_arg_params]
        avg_params[k] = avg.average_arrays(arrays, weights)

    for k in all_arg_params[0]:
        assert all_arg_params[0][k].shape == exp_output[k].shape
        assert mx.nd.sum(avg_params[k] - exp_output[k]) < 1e-20
