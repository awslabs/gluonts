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

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
)

import numpy as np
import pandas as pd
from autogluon import TabularPrediction as task
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import Localizer, Predictor
from gluonts.time_feature import (
    get_lags_for_frequency,
)

if TYPE_CHECKING:  # avoid circular import
    from gluonts.model.estimator import Estimator

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


def get_features_dataframe(
    series: pd.Series, lags: List[int] = []
) -> pd.DataFrame:
    # TODO allow customizing what features to use

    cal = calendar()
    holidays = cal.holidays(start=series.index.min(), end=series.index.max())

    time_features = {
        "year": series.index.year,
        "month_of_year": series.index.month,
        "day_of_week": series.index.dayofweek,
        "hour_of_day": series.index.hour,
        "holiday_indicator": series.index.isin(holidays),
    }

    lag_values = {f"lag_{idx}": series.shift(idx).values for idx in lags}

    columns = {**time_features, **lag_values, "target": series.values}

    return pd.DataFrame(columns, index=series.index)


class TabularPredictor(Predictor):
    def __init__(
        self,
        ag_model,
        freq: str,
        prediction_length: int,
        lags: List[int],
    ) -> None:
        assert all(l >= 1 for l in lags)

        self.ag_model = ag_model
        self.freq = freq
        self.prediction_length = prediction_length
        self.lags = lags
        self.auto_regression = (
            False if not lags else self.prediction_length > min(self.lags)
        )

    def predict(self, dataset: Iterable[Dict]) -> Iterator[SampleForecast]:
        for entry in dataset:
            series = to_pandas(entry)

            forecast_index = pd.date_range(
                series.index[-1] + series.index.freq,
                freq=series.index.freq,
                periods=self.prediction_length,
            )

            # TODO refactor below here

            if not self.auto_regression:
                # do all predictions at once
                forecast_series = pd.Series(
                    [None] * len(forecast_index),
                    index=forecast_index,
                )
                series = series.append(forecast_series)
                df = get_features_dataframe(series, self.lags).loc[
                    forecast_index
                ]
                ag_output = self.ag_model.predict(df)

            else:
                # do predictions one step at a time
                ag_output = np.array([])
                for k in range(len(forecast_index)):
                    step_index = forecast_index[k : k + 1]
                    step_series = pd.Series([None], index=step_index)
                    series = series.append(step_series)
                    df = get_features_dataframe(series, self.lags).loc[
                        step_index
                    ]
                    step_ag_output = self.ag_model.predict(df)
                    series[step_index] = step_ag_output
                    ag_output = np.append(ag_output, step_ag_output)

            yield self.to_forecast(
                ag_output,
                forecast_index[0],
                item_id=entry.get(FieldName.ITEM_ID, None),
            )

    def to_forecast(
        self, ag_output, start_timestamp, item_id=None
    ) -> Iterator[SampleForecast]:
        samples = ag_output.reshape((1, self.prediction_length))
        sample = SampleForecast(
            freq=self.freq,
            start_date=pd.Timestamp(start_timestamp, freq=self.freq),
            item_id=item_id,
            samples=samples,
        )
        return sample


class TabularEstimator(Estimator):
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        lags: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.task = task
        self.freq = freq
        self.prediction_length = prediction_length
        default_kwargs = {
            "eval_metric": "mean_absolute_error",
            "excluded_model_types": ["KNN", "XT", "RF"],
            "presets": [
                "high_quality_fast_inference_only_refit",
                "optimize_for_deployment",
            ],
        }
        self.kwargs = {**default_kwargs, **kwargs}
        self.lags = (
            lags if lags is not None else (get_lags_for_frequency(self.freq))
        )

    def train(self, training_data: Dataset) -> TabularPredictor:
        dfs = [
            get_features_dataframe(
                series=to_pandas(entry),
                lags=self.lags,
            )
            for entry in training_data
        ]
        df = pd.concat(dfs)
        ag_model = self.task.fit(
            df, label="target", problem_type="regression", **self.kwargs
        )
        return TabularPredictor(
            ag_model=ag_model,
            freq=self.freq,
            prediction_length=self.prediction_length,
            lags=self.lags,
        )


def LocalTabularPredictor(*args, **kwargs) -> Localizer:
    return Localizer(TabularEstimator(*args, **kwargs))
