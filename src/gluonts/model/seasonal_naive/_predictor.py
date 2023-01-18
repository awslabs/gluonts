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

from typing import Optional

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.util import forecast_start
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import (
    LastValueImputation,
    MissingValueImputation,
)
from gluonts.time_feature import get_seasonality


class SeasonalNaivePredictor(RepresentablePredictor):
    """
    Seasonal naïve forecaster.

    For each time series :math:`y`, this predictor produces a forecast
    :math:`\\tilde{y}(T+k) = y(T+k-h)`, where :math:`T` is the forecast time,
    :math:`k = 0, ...,` `prediction_length - 1`, and :math:`h =`
    `season_length`.

    If `prediction_length > season_length`, then the season is repeated
    multiple times. If a time series is shorter than season_length, then the
    mean observed value is used as prediction.

    Parameters
    ----------
    freq
        Frequency of the input data
    prediction_length
        Number of time points to predict
    season_length
        Length of the seasonality pattern of the input data
    imputation_method
        The imputation method to use in case of missing values.
        Defaults to `LastValueImputation` which replaces each missing
        value with the last value that was not missing.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        season_length: Optional[int] = None,
        imputation_method: Optional[
            MissingValueImputation
        ] = LastValueImputation(),
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        assert (
            season_length is None or season_length > 0
        ), "The value of `season_length` should be > 0"

        self.prediction_length = prediction_length
        self.season_length = (
            season_length
            if season_length is not None
            else get_seasonality(freq)
        )
        self.imputation_method = imputation_method

    def predict_item(self, item: DataEntry) -> Forecast:
        target = np.asarray(item[FieldName.TARGET], np.float32)
        len_ts = len(target)
        forecast_start_time = forecast_start(item)

        assert (
            len_ts >= 1
        ), "all time series should have at least one data point"

        if np.isnan(target).any():
            target = target.copy()
            target = self.imputation_method(target)

        if len_ts >= self.season_length:
            indices = [
                len_ts - self.season_length + k % self.season_length
                for k in range(self.prediction_length)
            ]
            samples = target[indices].reshape((1, self.prediction_length))
        else:
            samples = np.full(
                shape=(1, self.prediction_length),
                fill_value=np.nanmean(target),
            )

        return SampleForecast(
            samples=samples,
            start_date=forecast_start_time,
            item_id=item.get("item_id", None),
        )
