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
import pytest

# First-party imports
from gluonts.trainer import model_averaging as ma


@pytest.mark.parametrize("weight", ["average", "exp-metric"])
def tst_model_averaging(weight):
    param_1 = {
        "arg1": mx.nd.array([[1, 2, 3], [1, 2, 3]]),
        "arg2": mx.nd.array([[1, 2], [1, 2]]),
    }
    param_2 = {
        "arg1": mx.nd.array([[1, 1, 1], [1, 1, 1]]),
        "arg2": mx.nd.array([[1, 1], [1, 1]]),
    }

    all_arg_params = [param_1, param_2]

    if weight == "average":
        weights = [0.5, 0.5]
        exp_output = {
            "arg1": 0.5 * mx.nd.array([[1, 2, 3], [1, 2, 3]])
            + 0.5 * mx.nd.array([[1, 1, 1], [1, 1, 1]]),
            "arg2": 0.5 * mx.nd.array([[1, 2], [1, 2]])
            + 0.5 * mx.nd.array([[1, 1], [1, 1]]),
        }
    elif weight == "exp-metric":
        weights = [0.25, 0.75]
        exp_output = {
            "arg1": 0.25 * mx.nd.array([[1, 2, 3], [1, 2, 3]])
            + 0.75 * mx.nd.array([[1, 1, 1], [1, 1, 1]]),
            "arg2": 0.25 * mx.nd.array([[1, 2], [1, 2]])
            + 0.75 * mx.nd.array([[1, 1], [1, 1]]),
        }
    else:
        raise ValueError("Unknown value for 'weight'.")

    avg_params = {}
    for k in all_arg_params[0]:
        arrays = [p[k] for p in all_arg_params]
        avg_params[k] = ma.average_arrays(arrays, weights)

    for k in all_arg_params[0]:
        assert all_arg_params[0][k].shape == exp_output[k].shape
        assert mx.nd.sum(avg_params[k] - exp_output[k]) < 1e-20
