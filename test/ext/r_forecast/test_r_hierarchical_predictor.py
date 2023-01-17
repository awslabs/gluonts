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
import itertools
import pytest

from gluonts.core import serde
from gluonts.dataset.util import forecast_start
from gluonts.evaluation import MultivariateEvaluator, backtest_metrics
from gluonts.model import SampleForecast
from gluonts.ext.r_forecast import (
    RHierarchicalForecastPredictor,
    R_IS_INSTALLED,
    RPY2_IS_INSTALLED,
    SUPPORTED_HIERARCHICAL_METHODS,
)


def get_test_input(
    supported_hierarchical_methods: List[int],
    supported_base_forecast_methods: List[int],
    nonnegative_options: List[bool],
    levels: List[int],
    covariance_options: List[str],
    algorithm_options: List[str],
) -> List[Tuple[Union[str, int]]]:
    """
    We compute the combinations that we want for testing.
    """

    test_input = []
    test_input_constant = {
        "supported_base_forecast_methods": supported_base_forecast_methods,
        "nonnegative_options": nonnegative_options,
    }
    for hierarchical_method in supported_hierarchical_methods:
        if "mint" in hierarchical_method:
            test_input_method_dependant = {
                "levels": [None],
                "covariance_options": covariance_options,
                "algorithm_options": algorithm_options,
            }
        elif "middle_out" in hierarchical_method:
            test_input_method_dependant = {
                "levels": levels,
                "covariance_options": [None],
                "algorithm_options": [None],
            }
        else:
            test_input_method_dependant = {
                "levels": [None],
                "covariance_options": [None],
                "algorithm_options": [None],
            }

        test_input_params = {
            "hierarchical_method": [hierarchical_method],
            **test_input_constant,
            **test_input_method_dependant,
        }
        test_input.extend(list(itertools.product(*test_input_params.values())))

    return test_input


# conditionally skip these tests if `R` and `rpy2` are not installed
if not R_IS_INSTALLED or not RPY2_IS_INSTALLED:
    skip_message = "Skipping test because `R` and `rpy2` are not installed!"
    pytest.skip(skip_message, allow_module_level=True)

TOLERANCE = 4.0
SUPPORTED_BASE_FORECAST_METHODS = ["ets", "arima"]
LEVELS = [1, 2]
NONNEGATIVE = [True, False]
COVARIANCE = ["shr", "sam"]
ALGORITHM = ["lu", "cg"]

test_input = get_test_input(
    SUPPORTED_HIERARCHICAL_METHODS,
    SUPPORTED_BASE_FORECAST_METHODS,
    NONNEGATIVE,
    LEVELS,
    COVARIANCE,
    ALGORITHM,
)


@pytest.mark.parametrize(
    "method_name, fmethod, nonnegative, level, covariance, algorithm",
    test_input,
)
def test_forecasts(
    sine7, method_name, fmethod, nonnegative, level, covariance, algorithm
):
    train_datasets = sine7(nonnegative=nonnegative)
    prediction_length = 10

    (train_dataset, test_dataset, metadata) = (
        train_datasets.train,
        train_datasets.test,
        train_datasets.metadata,
    )

    freq = metadata.freq
    nodes = metadata.nodes
    target_dim, num_bottom_ts = train_datasets.metadata.S.shape

    params = dict(
        freq=freq,
        prediction_length=prediction_length,
        is_hts=True,
        target_dim=target_dim,
        num_bottom_ts=num_bottom_ts,
        nodes=nodes,
        method_name=method_name,
        fmethod=fmethod,
        nonnegative=nonnegative,
        covariance=covariance,
        algorithm=algorithm,
        level=level,
    )

    predictor = RHierarchicalForecastPredictor(**params)
    predictions = list(predictor.predict(train_dataset))

    assert all(
        isinstance(prediction, SampleForecast) for prediction in predictions
    )

    assert all(prediction.freq == freq for prediction in predictions)

    assert all(
        prediction.prediction_length == prediction_length
        for prediction in predictions
    )

    assert all(
        prediction.start_date == forecast_start(data)
        for data, prediction in zip(train_dataset, predictions)
    )

    agg_metrics, item_metrics = backtest_metrics(
        test_dataset=test_dataset,
        predictor=predictor,
        evaluator=MultivariateEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
    )

    assert agg_metrics["mean_wQuantileLoss"] < TOLERANCE
    assert agg_metrics["NRMSE"] < TOLERANCE
    assert agg_metrics["RMSE"] < TOLERANCE


def test_r_predictor_serialization():
    predictor = RHierarchicalForecastPredictor(
        freq="1D",
        prediction_length=3,
        is_hts=True,
        target_dim=7,
        num_bottom_ts=4,
        nodes=[2, [2] * 2],
        nonnegative=False,
        method_name=SUPPORTED_HIERARCHICAL_METHODS[0],
        fmethod="ets",
    )
    assert predictor == serde.decode(serde.encode(predictor))
