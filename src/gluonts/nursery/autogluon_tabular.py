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

from typing import TYPE_CHECKING, Callable, Dict, Iterable, Iterator

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

if TYPE_CHECKING:  # avoid circular import
    from gluonts.model.estimator import Estimator

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


def get_prediction_dataframe(series):
    hour_of_day = series.index.hour
    month_of_year = series.index.month
    day_of_week = series.index.dayofweek
    year_idx = series.index.year
    target = series.values
    cal = calendar()
    holidays = cal.holidays(start=series.index.min(), end=series.index.max())
    df = pd.DataFrame(
        zip(
            year_idx,
            month_of_year,
            day_of_week,
            hour_of_day,
            series.index.isin(holidays),
            target,
        ),
        columns=[
            "year_idx",
            "month_of_year",
            "day_of_week",
            "hour_of_day",
            "holiday",
            "target",
        ],
    )
    convert_type = {x: "category" for x in df.columns.values[:4]}
    df = df.astype(convert_type)
    return df


class TabularPredictor(Predictor):
    def __init__(
        self,
        ag_model,
        freq: str,
        prediction_length: int,
    ) -> None:
        self.ag_model = ag_model  # task?
        self.freq = freq
        self.prediction_length = prediction_length

    def predict(self, dataset: Iterable[Dict]) -> Iterator[SampleForecast]:
        for entry in dataset:
            ts = to_pandas(entry)
            start = ts.index[-1] + pd.tseries.frequencies.to_offset(self.freq)
            start_timestamp = pd.Timestamp(start, freq=self.freq)
            future_entry = {
                "start": start_timestamp,
                "target": np.array([None] * self.prediction_length),
            }
            future_ts = to_pandas(future_entry)
            df = get_prediction_dataframe(future_ts)
            ag_output = self.ag_model.predict(df)
            yield self.to_forecast(
                ag_output, start_timestamp, entry.get(FieldName.ITEM_ID, None)
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
    def __init__(self, freq: str, prediction_length: int, **kwargs) -> None:
        super().__init__()
        self.task = task
        self.freq = freq
        self.prediction_length = prediction_length
        default_kwargs = {
            "excluded_model_types": ["KNN", "XT", "RF"],
            "presets": [
                "high_quality_fast_inference_only_refit",
                "optimize_for_deployment",
            ],
            "eval_metric": "mean_absolute_error",
        }
        self.kwargs = {**default_kwargs, **kwargs}

    def train(self, training_data: Dataset) -> TabularPredictor:
        dfs = [
            get_prediction_dataframe(to_pandas(entry))
            for entry in training_data
        ]
        df = pd.concat(dfs)

        ag_model = self.task.fit(
            df, label="target", problem_type="regression", **self.kwargs
        )

        return TabularPredictor(ag_model, self.freq, self.prediction_length)


def LocalTabularPredictor(*args, **kwargs) -> Localizer:
    return Localizer(TabularEstimator(*args, **kwargs))
