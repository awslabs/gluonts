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

from gluonts.core import serde
from gluonts.dataset.repository import datasets
from gluonts.dataset.util import forecast_start, to_pandas
from gluonts.evaluation import Evaluator, backtest_metrics
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.model.r_forecast import (
    RForecastPredictor,
    R_IS_INSTALLED,
    RPY2_IS_INSTALLED,
    UNIVARIATE_QUANTILE_FORECAST_METHODS,
    SUPPORTED_UNIVARIATE_METHODS,
)


# conditionally skip these tests if `R` and `rpy2` are not installed
if not R_IS_INSTALLED or not RPY2_IS_INSTALLED:
    skip_message = "Skipping test because `R` and `rpy2` are not installed!"
    pytest.skip(msg=skip_message, allow_module_level=True)


TOLERANCE = 1e-6


@pytest.mark.parametrize("method_name", SUPPORTED_UNIVARIATE_METHODS)
def test_forecasts(method_name):
    if method_name == "mlp":
        # https://stackoverflow.com/questions/56254321/error-in-ifncol-matrix-rep-argument-is-of-length-zero
        # https://cran.r-project.org/web/packages/neuralnet/index.html
        #   published before the bug fix: https://github.com/bips-hb/neuralnet/pull/21
        # The issue is still open on nnfor package: https://github.com/trnnick/nnfor/issues/8
        # TODO: look for a workaround.
        pytest.xfail(
            "MLP currently does not work because "
            "the `neuralnet` package is not yet updated with a known bug fix in ` bips-hb/neuralnet`"
        )

    dataset = datasets.get_dataset("constant")

    (train_dataset, test_dataset, metadata) = (
        dataset.train,
        dataset.test,
        dataset.metadata,
    )

    freq = metadata.freq
    prediction_length = metadata.prediction_length

    params = dict(
        freq=freq, prediction_length=prediction_length, method_name=method_name
    )

    predictor = RForecastPredictor(**params)
    predictions = list(predictor.predict(train_dataset))

    forecast_type = (
        QuantileForecast
        if method_name in UNIVARIATE_QUANTILE_FORECAST_METHODS
        else SampleForecast
    )
    assert all(
        isinstance(prediction, forecast_type) for prediction in predictions
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

    evaluator = Evaluator()
    agg_metrics, item_metrics = backtest_metrics(
        test_dataset=test_dataset,
        predictor=predictor,
        evaluator=evaluator,
    )
    assert agg_metrics["mean_wQuantileLoss"] < TOLERANCE
    assert agg_metrics["NRMSE"] < TOLERANCE
    assert agg_metrics["RMSE"] < TOLERANCE

    trunc_length = prediction_length

    predictor = RForecastPredictor(**params, trunc_length=trunc_length)
    predictions = list(predictor.predict(train_dataset))

    assert all(
        prediction.start_date == to_pandas(data).index[-1] + 1
        for data, prediction in zip(train_dataset, predictions)
    )


def test_r_predictor_serialization():
    predictor = RForecastPredictor(freq="1D", prediction_length=3)
    assert predictor == serde.decode(serde.encode(predictor))
