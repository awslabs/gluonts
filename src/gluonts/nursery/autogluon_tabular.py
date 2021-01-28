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


def get_prediction_dataframe(series, prediction_length, use_lag, context_length, future_series=None, train=False):
    if use_lag and not train:
        series = series.append(future_series)
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

    cache = [None] * (context_length + prediction_length) + list(series.values)
    col = []
    total = len(df)
    for i in range(context_length):
        col.append('lag' + str(i))
        df['lag' + str(i)] = cache[i:i + total]
    return df if train else df[-prediction_length:]


class TabularPredictor(Predictor):
    def __init__(self, ag_model, freq: str, prediction_length: int, lags: list, use_lag=False) -> None:
        self.ag_model = ag_model
        self.freq = freq
        self.prediction_length = prediction_length
        self.use_lag = use_lag
        self.lags = lags
        self.auto_regression = False if ((not self.use_lag) or self.lags[-1] < self.prediction_length) else True
        self.context_length = len(lags)

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
            df = get_prediction_dataframe(ts, self.prediction_length, self.use_lag, self.context_length, future_ts, train=False)
            if not self.auto_regression:
                ag_output = self.ag_model.predict(df)
            else:
                train_len = len(ts.values)
                ag_output = np.array([])
                cache = [None] * self.context_length + list(ts.values)
                idx = df.index[0]
                context_cols = df.columns[-self.context_length:]
                for i in range(self.prediction_length):
                    row_to_predict = df.iloc[[i], :]
                    context = cache[(train_len + i): (train_len + self.context_length + i)]
                    df.loc[[i + idx], context_cols] = context
                    cur_output = self.ag_model.predict(row_to_predict)
                    ag_output = np.append(ag_output, cur_output)
                    cache.extend(cur_output)
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
    def __init__(self, freq: str, prediction_length: int, use_lag=False, lags=[], **kwargs) -> None:
        super().__init__()
        self.task = task
        self.freq = freq
        self.prediction_length = prediction_length
        default_kwargs = {
            "eval_metric": "mean_absolute_error",
            "excluded_model_types": ["KNN", "XT", "RF"],
            "presets": [
                "high_quality_fast_inference_only_refit", "optimize_for_deployment"
            ],
        }
        self.kwargs = {**default_kwargs, **kwargs}
        self.use_lag = use_lag
        if not self.use_lag:
            self.lags = []
        else:
            self.lags = sorted(list(set(lags))) if lags else list(get_lags_for_frequency(self.freq))

    def train(self, training_data: Dataset) -> TabularPredictor:
        # every time there is only one time series passed
        # list(training_data)[0] is essentially getting the only time series
        dfs = [
            get_prediction_dataframe(series=to_pandas(entry),
                                     prediction_length=self.prediction_length,
                                     use_lag=self.use_lag,
                                     context_length=len(self.lags), train=True)
            for entry in training_data
        ]
        df = pd.concat(dfs)
        ag_model = self.task.fit(df, label="target", problem_type="regression", output_directory="Eval2/electricity30",
                                 **self.kwargs)
        return TabularPredictor(ag_model=ag_model, freq=self.freq, prediction_length=self.prediction_length,
                                use_lag=self.use_lag, lags=self.lags)


def LocalTabularPredictor(*args, **kwargs) -> Localizer:
    return Localizer(TabularEstimator(*args, **kwargs))

