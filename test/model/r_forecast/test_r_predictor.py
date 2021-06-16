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

from gluonts.core import serde
from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.model.r_forecast import (
    RForecastPredictor,
    R_IS_INSTALLED,
    RPY2_IS_INSTALLED,
    SUPPORTED_METHODS,
    QUANTILE_FORECAST_METHODS,
)
from gluonts.support.pandas import forecast_start


# conditionally skip these tests if `R` and `rpy2` are not installed
if not R_IS_INSTALLED or not RPY2_IS_INSTALLED:
    skip_message = "Skipping test because `R` and `rpy2` are not installed!"
    pytest.skip(msg=skip_message, allow_module_level=True)


@pytest.mark.parametrize("method_name", SUPPORTED_METHODS)
def test_forecast_structure(method_name):
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

    freq = "1D"
    prediction_length = 10
    params = dict(
        freq=freq, prediction_length=prediction_length, method_name=method_name
    )

    dataset = ListDataset(
        data_iter=[
            {"start": "2017-01-01", "target": np.array([1.0] * 3)},
            {"start": "2007-09-28", "target": np.array([2.0] * 3)},
            {"start": "2020-07-20", "target": np.array([3.0] * 3)},
            {"start": "1947-08-15", "target": np.array([4.0] * 3)},
        ],
        freq=params["freq"],
    )

    predictor = RForecastPredictor(**params)
    predictions = list(predictor.predict(dataset))

    forecast_type = (
        QuantileForecast
        if method_name in QUANTILE_FORECAST_METHODS
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
        for data, prediction in zip(dataset, predictions)
    )


def test_r_predictor_serialization():
    predictor = RForecastPredictor(freq="1D", prediction_length=3)
    assert predictor == serde.decode(serde.encode(predictor))
