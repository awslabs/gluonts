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

import pytest

from gluonts.model.n_beats import NBEATSEnsembleEstimator, NBEATSEstimator


@pytest.fixture()
def args(dsinfo):
    common_hp = dict(
        ctx="cpu", epochs=1, hybridize=True, num_batches_per_epoch=1,
    )
    return {
        "generic": {
            "estimator": NBEATSEstimator,
            "args": {"num_stacks": 2},
            **common_hp,
        },
        "interpretable": {
            "estimator": NBEATSEstimator,
            "args": {
                "num_stacks": 2,
                "num_blocks": [3],
                "widths": [256, 2048],
                "sharing": [True],
                "stack_types": ["T", "S"],
                **common_hp,
            },
        },
        "ensemble": {
            "estimator": NBEATSEnsembleEstimator,
            "args": {
                "meta_context_length": [2 * dsinfo.prediction_length],
                "meta_loss_function": ["MAPE"],
                "meta_bagging_size": 2,
                "num_stacks": 3,
                **common_hp,
            },
        },
    }


@pytest.fixture(params=["generic", "interpretable", "ensemble"])
def name(request):
    return request.param


@pytest.fixture()
def estimator_config(args, name):
    config = args[name]
    return name, config["estimator"], config["args"]


# tests are too slow
@pytest.mark.skip("Skipping slow test.")
@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(accuracy_test, estimator_config, hybridize):
    Estimator, hyperparameters = estimator_config[1:]
    hyperparameters.update(num_batches_per_epoch=200, hybridize=hybridize)

    accuracy_test(Estimator, hyperparameters, accuracy=0.3)


def test_repr(repr_test, estimator_config):
    repr_test(*estimator_config[1:])


def test_serialize(serialize_test, estimator_config):
    if estimator_config[0] == "generic":
        pytest.skip("Too slow.")
    serialize_test(*estimator_config[1:])
