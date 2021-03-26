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

import json
import sys
from distutils.util import strtobool
from typing import ContextManager

import numpy as np
import pytest

from gluonts.core.component import equals
from gluonts.dataset.common import FileDataset, ListDataset
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.trivial.mean import MeanPredictor
from gluonts.shell.env import ServeEnv, TrainEnv
from gluonts.shell.train import run_train_and_test

try:
    from gluonts.shell.serve import Settings
    from gluonts.shell.serve.util import jsonify_floats
    from gluonts.testutil import shell as testutil
except ImportError:
    if sys.platform != "win32":
        raise

    # gunicorn doesn't work on windows, so we skip these tests
    pytestmark = pytest.mark.skip


context_length = 5
prediction_length = 6
num_samples = 4


@pytest.fixture(scope="function")  # type: ignore
def train_env(listify_dataset) -> ContextManager[TrainEnv]:
    hyperparameters = {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "num_samples": num_samples,
        "listify_dataset": listify_dataset,
        "num_workers": 3,
        "num_prefetch": 4,
        "shuffle_buffer_length": 256,
        "epochs": 3,
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
        "test_quantiles": [0.25, 0.75],
    }
    with testutil.temporary_train_env(hyperparameters, "constant") as env:
        yield env


@pytest.fixture(scope="function")  # type: ignore
def static_server(
    train_env: TrainEnv,
) -> ContextManager["testutil.ServerFacade"]:
    predictor = MeanPredictor.from_hyperparameters(**train_env.hyperparameters)
    predictor.serialize(train_env.path.model)

    serve_env = ServeEnv(train_env.path.base)
    settings = Settings(sagemaker_server_port=testutil.free_port())
    with testutil.temporary_server(serve_env, None, settings) as server:
        yield server


@pytest.fixture(scope="function")  # type: ignore
def dynamic_server(
    train_env: TrainEnv,
) -> ContextManager["testutil.ServerFacade"]:
    serve_env = ServeEnv(train_env.path.base)
    settings = Settings(sagemaker_server_port=testutil.free_port())
    with testutil.temporary_server(
        serve_env, MeanPredictor, settings
    ) as server:
        yield server


@pytest.fixture
def batch_transform(monkeypatch, train_env):
    monkeypatch.setenv("SAGEMAKER_BATCH", "true")

    inference_config = {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "num_samples": num_samples,
        "output_types": ["mean", "samples"],
        "quantiles": [],
        **train_env.hyperparameters,
    }

    monkeypatch.setenv("INFERENCE_CONFIG", json.dumps(inference_config))
    monkeypatch.setenv("GLUONTS_FORWARD_FIELDS", json.dumps(["foo"]))
    return inference_config


@pytest.mark.parametrize("listify_dataset", ["yes", "no"])
def test_listify_dataset(train_env: TrainEnv, listify_dataset):
    for dataset in train_env.datasets.values():
        if listify_dataset == "yes":
            assert isinstance(dataset, ListDataset)
        else:
            assert isinstance(dataset, FileDataset)


@pytest.mark.parametrize("listify_dataset", ["yes", "no"])
@pytest.mark.parametrize("forecaster_type", [MeanPredictor, MQCNNEstimator])
def test_train_shell(train_env: TrainEnv, caplog, forecaster_type) -> None:
    run_train_and_test(env=train_env, forecaster_type=forecaster_type)

    if forecaster_type == MeanPredictor:
        for _, _, line in caplog.record_tuples:
            if "#test_score (local, QuantileLoss" in line:
                assert line.endswith("0.0")
            if "local, wQuantileLoss" in line:
                assert line.endswith("0.0")
            if "local, Coverage" in line:
                assert line.endswith("0.0")
            if "MASE" in line or "MSIS" in line:
                assert line.endswith("nan")
            if "abs_target_sum" in line:
                assert line.endswith("270.0")


@pytest.mark.parametrize("listify_dataset", ["yes", "no"])
def test_server_shell(
    train_env: TrainEnv, static_server: "testutil.ServerFacade", caplog
) -> None:
    execution_parameters = static_server.execution_parameters()

    assert "BatchStrategy" in execution_parameters
    assert "MaxConcurrentTransforms" in execution_parameters
    assert "MaxPayloadInMB" in execution_parameters

    assert execution_parameters["BatchStrategy"] == "SINGLE_RECORD"
    assert execution_parameters["MaxPayloadInMB"] == 6

    configuration = {
        "num_samples": 1,  # FIXME: this is ignored
        "output_types": ["mean", "samples"],
        "quantiles": [],
    }

    for entry in train_env.datasets["train"]:
        forecast = static_server.invocations([entry], configuration)[0]

        for output_type in configuration["output_types"]:
            assert output_type in forecast

        act_mean = np.array(forecast["mean"])
        act_samples = np.array(forecast["samples"])

        mean = np.mean(entry["target"])

        exp_mean_shape = (prediction_length,)
        exp_samples_shape = (num_samples, prediction_length)

        exp_mean = mean * np.ones(shape=(prediction_length,))
        exp_samples = mean * np.ones(shape=exp_samples_shape)

        assert exp_mean_shape == act_mean.shape
        assert exp_samples_shape == act_samples.shape
        assert equals(exp_mean, act_mean)
        assert equals(exp_samples, act_samples)


@pytest.mark.parametrize("listify_dataset", ["yes", "no"])
def test_dynamic_shell(
    train_env: TrainEnv, dynamic_server: "testutil.ServerFacade", caplog
) -> None:
    execution_parameters = dynamic_server.execution_parameters()

    assert "BatchStrategy" in execution_parameters
    assert "MaxConcurrentTransforms" in execution_parameters
    assert "MaxPayloadInMB" in execution_parameters

    assert execution_parameters["BatchStrategy"] == "SINGLE_RECORD"
    assert execution_parameters["MaxPayloadInMB"] == 6

    configuration = {
        "num_eval_samples": 1,  # FIXME: this is ignored
        "output_types": ["mean", "samples"],
        "quantiles": [],
        **train_env.hyperparameters,
    }

    for entry in train_env.datasets["train"]:
        forecast = dynamic_server.invocations([entry], configuration)[0]

        for output_type in configuration["output_types"]:
            assert output_type in forecast

        act_mean = np.array(forecast["mean"])
        act_samples = np.array(forecast["samples"])

        mean = np.mean(entry["target"])

        exp_mean_shape = (prediction_length,)
        exp_samples_shape = (num_samples, prediction_length)

        exp_mean = mean * np.ones(shape=(prediction_length,))
        exp_samples = mean * np.ones(shape=exp_samples_shape)

        assert exp_mean_shape == act_mean.shape
        assert exp_samples_shape == act_samples.shape
        assert equals(exp_mean, act_mean)
        assert equals(exp_samples, act_samples)


@pytest.mark.parametrize("listify_dataset", ["yes", "no"])
def test_dynamic_batch_shell(
    batch_transform,
    train_env: TrainEnv,
    dynamic_server: "testutil.ServerFacade",
    caplog,
) -> None:
    execution_parameters = dynamic_server.execution_parameters()

    assert "BatchStrategy" in execution_parameters
    assert "MaxConcurrentTransforms" in execution_parameters
    assert "MaxPayloadInMB" in execution_parameters

    assert execution_parameters["BatchStrategy"] == "SINGLE_RECORD"
    assert execution_parameters["MaxPayloadInMB"] == 6

    for entry in train_env.datasets["train"]:
        entry["foo"] = 42
        forecast = dynamic_server.batch_invocations([entry])[0]

        for output_type in batch_transform["output_types"]:
            assert output_type in forecast

        assert forecast["foo"] == 42

        act_mean = np.array(forecast["mean"])
        act_samples = np.array(forecast["samples"])

        mean = np.mean(entry["target"])

        exp_mean_shape = (prediction_length,)
        exp_samples_shape = (num_samples, prediction_length)

        exp_mean = mean * np.ones(shape=(prediction_length,))
        exp_samples = mean * np.ones(shape=exp_samples_shape)

        assert exp_mean_shape == act_mean.shape
        assert exp_samples_shape == act_samples.shape
        assert equals(exp_mean, act_mean)
        assert equals(exp_samples, act_samples)


def test_as_json_dict_outputs_valid_json():
    non_compliant_json = {
        "a": float("nan"),
        "k": float("infinity"),
        "b": {
            "c": float("nan"),
            "d": "testing",
            "e": float("-infinity"),
            "f": float("infinity"),
            "g": {"h": float("nan")},
        },
    }

    with pytest.raises(ValueError):
        json.dumps(non_compliant_json, allow_nan=False)

    output_json = jsonify_floats(non_compliant_json)
    json.dumps(output_json, allow_nan=False)
