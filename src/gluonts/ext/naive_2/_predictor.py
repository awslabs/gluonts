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
import statsmodels.api as sm

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor


def seasonality_test(past_ts_data: np.array, season_length: int) -> bool:
    """
    Test the time series for seasonal patterns by performing a 90% auto-
    correlation test.

    For details, see:
    http://www.unic.ac.cy/test/wp-content/uploads/sites/2/2018/09/M4-Competitors-Guide.pdf

    Code based on:
    https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
    """
    critical_z_score = 1.645  # corresponds to 90% prediction interval
    if len(past_ts_data) < 3 * season_length:
        return False
    else:
        # calculate auto-correlation for lags up to season_length
        auto_correlations = sm.tsa.stattools.acf(
            past_ts_data, fft=False, nlags=season_length
        )
        auto_correlations[1:] = 2 * auto_correlations[1:] ** 2
        limit = (
            critical_z_score
            / np.sqrt(len(past_ts_data))
            * np.sqrt(np.cumsum(auto_correlations))
        )
        is_seasonal = (
            abs(auto_correlations[season_length]) > limit[season_length]
        )

    return is_seasonal


def naive_2(
    past_ts_data: np.ndarray,
    prediction_length: int,
    season_length: int,
) -> np.ndarray:
    """
    Make seasonality adjusted time series prediction.

    For details, see:
    http://www.unic.ac.cy/test/wp-content/uploads/sites/2/2018/09/M4-Competitors-Guide.pdf

    Code based on:
    https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
    """
    assert season_length > 0, "The value of `season_length` should be > 0"

    has_seasonality = False

    if season_length > 1:
        has_seasonality = seasonality_test(past_ts_data, season_length)

    # it has seasonality, then calculate the multiplicative seasonal component
    if has_seasonality:
        # TODO: think about maybe only using past_ts_data[- max
        # (5*season_length, 2*prediction_length):] for speedup
        seasonal_decomposition = sm.tsa.seasonal_decompose(
            x=past_ts_data, period=season_length, model="multiplicative"
        ).seasonal
        seasonality_normed_context = past_ts_data / seasonal_decomposition

        last_period = seasonal_decomposition[-season_length:]
        num_required_periods = (prediction_length // len(last_period)) + 1

        multiplicative_seasonal_component = np.tile(
            last_period, num_required_periods
        )[:prediction_length]
    else:
        seasonality_normed_context = past_ts_data
        multiplicative_seasonal_component = np.ones(
            prediction_length
        )  # i.e. no seasonality component

    # calculate naive forecast: (last value prediction_length times)
    naive_forecast = (
        np.ones(prediction_length) * seasonality_normed_context[-1]
    )

    forecast = np.mean(naive_forecast) * multiplicative_seasonal_component

    return forecast


class Naive2Predictor(RepresentablePredictor):
    """
    Naïve 2 forecaster as described in the M4 Competition Guide:
    http://www.unic.ac.cy/test/wp-content/uploads/sites/2/2018/09/M4-Competitors-Guide.pdf

    The Python analogue implementation to:
    https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R#L118

    Parameters
    ----------
    prediction_length
        Number of time points to predict
    season_length
        Length of the seasonality pattern of the input data
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        season_length: int,
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        assert season_length > 0, "The value of `season_length` should be > 0"

        self.prediction_length = prediction_length
        self.season_length = season_length

    def predict_item(self, item: DataEntry) -> Forecast:
        past_ts_data = item["target"]
        item_id = item.get("item_id", None)
        forecast_start_time = forecast_start(item)

        assert (
            len(past_ts_data) >= 1
        ), "all time series should have at least one data point"

        prediction = naive_2(
            past_ts_data, self.prediction_length, self.season_length
        )

        samples = np.array([prediction])

        return SampleForecast(
            samples=samples,
            start_date=forecast_start_time,
            item_id=item_id,
        )
