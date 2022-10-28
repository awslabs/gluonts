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
    levels: List[int],
) -> List[Tuple[Union[str, int]]]:
    """
    We compute the combinations that we want for testing. For middle_out we need the *level* taken as reference,
    whereas for top_down and bottom_up *level* is not required.
    """

    supported_hierarchical_methods_without_middle_out = []
    supported_hierarchical_methods_with_middle_out = []

    for e in supported_hierarchical_methods:
        if "middle_out" in e:
            supported_hierarchical_methods_with_middle_out.append(e)
        else:
            supported_hierarchical_methods_without_middle_out.append(e)

    test_input = list(
        itertools.product(
            supported_hierarchical_methods_without_middle_out,
            supported_base_forecast_methods,
            [None],
        )
    )

    test_input.extend(
        list(
            itertools.product(
                supported_hierarchical_methods_with_middle_out,
                supported_base_forecast_methods,
                levels,
            )
        )
    )

    return test_input


# conditionally skip these tests if `R` and `rpy2` are not installed
if not R_IS_INSTALLED or not RPY2_IS_INSTALLED:
    skip_message = "Skipping test because `R` and `rpy2` are not installed!"
    pytest.skip(msg=skip_message, allow_module_level=True)

TOLERANCE = 4.0
SUPPORTED_BASE_FORECAST_METHODS = ["ets", "arima"]
LEVELS = [1, 2]
test_input = get_test_input(
    SUPPORTED_HIERARCHICAL_METHODS, SUPPORTED_BASE_FORECAST_METHODS, LEVELS
)


# @pytest.mark.parametrize("method_name, fmethod, level", test_input)
# @pytest.mark.parametrize("fmethod", SUPPORTED_BASE_FORECAST_METHODS)
@pytest.mark.parametrize("fmethod", ["ets"])
@pytest.mark.parametrize("method_name", SUPPORTED_HIERARCHICAL_METHODS)
@pytest.mark.parametrize("covariance", ["shr"])#, "sam"])
@pytest.mark.parametrize("nonnegative", [True])
@pytest.mark.parametrize("algorithm", ["cg"])
# @pytest.mark.parametrize("algorithm", ["lu", "cg", "chol"])
def test_forecasts(sine7, method_name, fmethod, covariance, nonnegative, algorithm):
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
        nonnegative=False,
        method_name=method_name,
        fmethod=fmethod,
    )

    # if level:
    #     params["level"] = level

    if covariance:
        params["covariance"] = covariance

    if nonnegative:
        params["nonnegative"] = nonnegative

    if algorithm:
        params["algorithm"] = algorithm

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
