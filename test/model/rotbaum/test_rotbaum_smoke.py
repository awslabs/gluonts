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
from pathlib import Path

import numpy as np
import pytest
from toolz import first

from gluonts.model.rotbaum import TreeEstimator
from gluonts.model.rotbaum import TreePredictor
from gluonts.shell.env import ServeEnv
from gluonts.shell.serve import Settings
from gluonts.testutil import shell as testutil
from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features


# TODO: Add support for categorical and dynamic features.


@pytest.mark.parametrize(
    "datasets",
    [
        # No features
        make_dummy_datasets_with_features(),
        # Single static categorical feature
        make_dummy_datasets_with_features(cardinality=[5]),
        # Multiple static categorical features
        make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        # Multiple static categorical features
        make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        # Single dynamic real feature
        make_dummy_datasets_with_features(num_feat_dynamic_real=1),
        # Multiple dynamic real feature
        make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        # Multiple dynamic real feature
        make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        # Both static categorical and dynamic real features
        make_dummy_datasets_with_features(
            cardinality=[3, 10, 42], num_feat_dynamic_real=3
        ),
        # Both static categorical and dynamic real features
        make_dummy_datasets_with_features(
            cardinality=[3, 10, 42], num_feat_dynamic_real=3
        ),
    ],
)
def test_rotbaum_quantile_regression_smoke(datasets):
    dataset_train, dataset_test = datasets
    hps = {
        "freq": "D",
        "prediction_length": 3,
        "context_length": 3,
        "quantiles": [0.1, 0.5, 0.9],
        "method": "QuantileRegression",
    }
    estimator = TreeEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)
    quantile_preds = np.array(
        [
            [forecast.quantile(quantile) for quantile in hps["quantiles"]]
            for forecast in forecasts
        ]
    )
    assert np.all(quantile_preds == np.sort(quantile_preds, axis=1))  # Is
    # quantile crossing resolved?


@pytest.mark.parametrize("methods", ["QRX", "QuantileRegression", "QRF"])
def test_rotbaum_serde(methods):
    dataset_train, dataset_test = make_dummy_datasets_with_features(
        cardinality=[3, 10, 42], num_feat_dynamic_real=3
    )
    hps = {
        "freq": "D",
        "prediction_length": 3,
        "context_length": 3,
        "quantiles": [0.1, 0.5, 0.9],
        "method": methods,
    }
    estimator = TreeEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train)
    predictor.serialize(Path("/tmp/"))
    another_predictor = TreePredictor.deserialize(Path("/tmp/"))
    assert another_predictor.preprocess_object is not None
    assert len(another_predictor.model_list) == 3


@pytest.mark.parametrize("methods", ["QRX", "QRF"])
def test_mean_forecasts_for_non_supported_methods(methods):
    dataset_train, dataset_test = make_dummy_datasets_with_features()
    hps = {
        "freq": "D",
        "prediction_length": 3,
        "context_length": 3,
        "quantiles": [0.1, 0.5, 0.9],
        "method": methods,
    }
    estimator = TreeEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train)
    forecasts = list(predictor.predict(dataset_test))
    with pytest.raises(AssertionError) as assertInfo:
        for forecast in forecasts:
            forecast.mean
    assert "mean is only supported by QuantileRegression method" in str(
        assertInfo.value
    )


@pytest.mark.parametrize("methods", ["QuantileRegression"])
def test_mean_forecasts_for_supported_methods(methods):
    dataset_train, dataset_test = make_dummy_datasets_with_features()
    hps = {
        "freq": "D",
        "prediction_length": 3,
        "context_length": 3,
        "quantiles": [0.1, 0.5, 0.9],
        "method": methods,
    }
    estimator = TreeEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train)
    forecasts = list(predictor.predict(dataset_test))
    assert all(
        np.array_equal(forecast.mean, forecast.quantile(0.5))
        for forecast in forecasts
    )


def test_rotbaum_inference_server() -> None:
    hps = {
        "freq": "1H",
        "prediction_length": 3,
        "quantiles": [0.1, 0.5, 0.9],
        "method": "QuantileRegression",
    }
    with testutil.temporary_train_env(hps, "constant") as train_env:
        estimator = TreeEstimator.from_inputs(
            train_env.datasets["train"], **train_env.hyperparameters
        )
        predictor = estimator.train(train_env.datasets["train"])
        predictor.serialize(train_env.path.model)
        assert predictor.explain()
        serve_env = ServeEnv(train_env.path.base)
        settings = Settings(
            sagemaker_server_port=testutil.free_port(), model_server_workers=1
        )
        with testutil.temporary_server(serve_env, None, settings) as server:
            configuration = {
                "num_samples": 1,  # doesn't matter since the predictor is not sampling based
                "output_types": ["mean", "quantiles"],
                "quantiles": ["0.1", "0.5", "0.9"],
            }

            entry = first(train_env.datasets["train"])
            for forecast in server.invocations([entry], configuration):
                assert all(
                    output_type in forecast
                    for output_type in configuration["output_types"]
                )
