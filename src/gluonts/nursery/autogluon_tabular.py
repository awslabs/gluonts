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
from gluonts.itertools import batcher
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import Localizer, Predictor
from gluonts.time_feature import get_lags_for_frequency


def get_features_dataframe(
    series: pd.Series,
    lags: List[int] = [],
    past_data: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Constructs a DataFrame of features for a given Series.

    Features include some date-time features (like hour-of-day, day-of-week, ...) and
    lagged values from the series itself. Lag indices are specified by `lags`, while
    previous data can be specified by `past_data`: the latter allows to get lags also
    for the initial values of the series.

    Parameters
    ----------
    series
        Series on which features should be computed.
    lags
        Indices of lagged observations to include as features.
    past_data
        Prior data, to be used to compute lagged observations.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the features. This has the same index as `series`.
    """
    # TODO allow customizing what features to use
    # TODO check if anything can be optimized here

    assert past_data is None or series.index.freq == past_data.index.freq
    assert past_data is None or series.index[0] > past_data.index[-1]

    cal = calendar()
    holidays = cal.holidays(start=series.index.min(), end=series.index.max())
    time_features = {
        "year": series.index.year,
        "month_of_year": series.index.month,
        "day_of_week": series.index.dayofweek,
        "hour_of_day": series.index.hour,
        "holiday_indicator": series.index.isin(holidays),
    }

    all_data = (
        series
        if past_data is None
        else past_data.append(series).asfreq(series.index.freq)
    )
    lag_values = {
        f"lag_{idx}": all_data.shift(idx)[series.index].values for idx in lags
    }

    columns = {**time_features, **lag_values, "target": series.values}

    return pd.DataFrame(columns, index=series.index)


class TabularPredictor(Predictor):
    def __init__(
        self,
        ag_model,
        freq: str,
        prediction_length: int,
        lags: List[int],
        batch_size: int = 32,
        dtype=np.float32,
    ) -> None:
        super().__init__(prediction_length=prediction_length, freq=freq)
        assert all(l >= 1 for l in lags)

        self.ag_model = ag_model
        self.lags = lags
        self.dtype = dtype

    @property
    def auto_regression(self) -> bool:
        return (
            False if not self.lags else self.prediction_length > min(self.lags)
        )

    def _to_forecast(
        self,
        ag_output: np.ndarray,
        start_timestamp: pd.Timestamp,
        item_id=None,
    ) -> Iterator[SampleForecast]:
        samples = ag_output.reshape((1, self.prediction_length))
        sample = SampleForecast(
            freq=self.freq,
            start_date=pd.Timestamp(start_timestamp, freq=self.freq),
            item_id=item_id,
            samples=samples,
        )
        return sample

    # serial prediction (both auto-regressive and not)
    # `auto_regression == False`: one call to Autogluon's `predict` per input time series
    # `auto_regression == True`: `prediction_length` calls to Autogluon's `predict` per input time series
    # really only useful for debugging, since this is generally slower than the batched versions (see below)
    def _predict_serial(
        self, dataset: Iterable[Dict], **kwargs
    ) -> Iterator[SampleForecast]:
        for entry in dataset:
            series = to_pandas(entry)

            forecast_index = pd.date_range(
                series.index[-1] + series.index.freq,
                freq=series.index.freq,
                periods=self.prediction_length,
            )

            forecast_series = pd.Series(
                [None] * len(forecast_index),
                index=forecast_index,
            )

            full_series = series.append(forecast_series)

            if not self.auto_regression:  # predict all at once
                df = get_features_dataframe(
                    forecast_series, self.lags, past_data=series
                )
                full_series[forecast_series.index] = self.ag_model.predict(df)

            else:  # predict step by step
                for idx in forecast_series.index:
                    df = get_features_dataframe(
                        forecast_series[idx:idx],
                        self.lags,
                        past_data=full_series[:idx][:-1],
                    )
                    full_series[idx] = self.ag_model.predict(df).item()

            yield self._to_forecast(
                full_series[forecast_index].values.astype(self.dtype),
                forecast_index[0],
                item_id=entry.get(FieldName.ITEM_ID, None),
            )

    # batch prediction (no auto-regression)
    # one call to Autogluon's `predict`
    def _predict_batch_one_shot(
        self, dataset: Iterable[Dict], **kwargs
    ) -> Iterator[SampleForecast]:
        # TODO clean up
        # TODO optimize
        dfs = []
        forecast_start_timestamps = []
        item_ids = []
        for entry in dataset:
            series = to_pandas(entry)
            forecast_start = series.index[-1] + series.index.freq
            forecast_index = pd.date_range(
                forecast_start,
                freq=series.index.freq,
                periods=self.prediction_length,
            )
            forecast_series = pd.Series(
                [None] * self.prediction_length,
                index=forecast_index,
            )
            dfs.append(
                get_features_dataframe(
                    forecast_series, self.lags, past_data=series
                )
            )
            forecast_start_timestamps.append(forecast_start)
            item_ids.append(entry.get(FieldName.ITEM_ID, None))

        df = pd.concat(dfs)
        output = self.ag_model.predict(df)
        for arr, forecast_start, item_id in zip(
            np.split(output, len(dfs)), forecast_start_timestamps, item_ids
        ):
            yield self._to_forecast(
                arr,
                forecast_start,
                item_id=item_id,
            )

    # batch prediction (auto-regressive)
    # `prediction_length` calls to Autogluon's `predict`
    def _predict_batch_autoreg(
        self, dataset: Iterable[Dict], **kwargs
    ) -> Iterator[SampleForecast]:
        # TODO clean up
        # TODO optimize
        batch_series = []
        batch_ids = []
        for entry in dataset:
            batch_series.append(to_pandas(entry))
            batch_ids.append(entry.get(FieldName.ITEM_ID, None))
        batch_forecast_indices = [
            pd.date_range(
                series.index[-1] + series.index.freq,
                freq=series.index.freq,
                periods=self.prediction_length,
            )
            for series in batch_series
        ]
        batch_full_series = [
            series.append(
                pd.Series(
                    [None] * self.prediction_length,
                    index=forecast_index,
                )
            )
            for series, forecast_index in zip(
                batch_series, batch_forecast_indices
            )
        ]

        output = np.zeros(
            (len(batch_series), self.prediction_length), dtype=self.dtype
        )

        for k in range(self.prediction_length):
            dfs = []
            for fs, idx in zip(batch_full_series, batch_forecast_indices):
                idx_k = idx[k]
                dfs.append(
                    get_features_dataframe(
                        fs[idx_k:idx_k],
                        self.lags,
                        past_data=fs[:idx_k][:-1],
                    )
                )
            df = pd.concat(dfs)
            out_k = self.ag_model.predict(df)
            output[:, k] = out_k
            for fs, idx, v in zip(
                batch_full_series, batch_forecast_indices, out_k
            ):
                fs.at[idx[k]] = v

        for arr, forecast_index, item_id in zip(
            output, batch_forecast_indices, batch_ids
        ):
            yield self._to_forecast(
                arr,
                forecast_index[0],
                item_id=item_id,
            )

    def _predict_batch(
        self, dataset: Iterable[Dict], batch_size: int, **kwargs
    ) -> Iterator[SampleForecast]:
        for batch in batcher(dataset, batch_size):
            yield from (
                self._predict_batch_one_shot(batch, **kwargs)
                if not self.auto_regression
                else self._predict_batch_autoreg(batch, **kwargs)
            )

    def predict(
        self,
        dataset: Iterable[Dict],
        batch_size: Optional[int] = 32,
        **kwargs,
    ) -> Iterator[SampleForecast]:
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is None:
            return self._predict_serial(dataset, **kwargs)
        else:
            return self._predict_batch(
                dataset, batch_size=batch_size, **kwargs
            )


class TabularEstimator(Estimator):
    """An estimator that trains an Autogluon Tabular model for time series
    forecasting.

    Additional keyword arguments to the constructor will be passed on to
    Autogluon Tabular's ``fit`` method used for training the model.

    Parameters
    ----------
    freq
        Frequency of the data to handle
    prediction_length
        Prediction length
    lags
        List of indices of the lagged observations to use as features. If
        None, this will be set automatically based on the frequency.
    batch_size
        Batch size of the resulting predictor; this is just used at prediction
        time, and does not affect training in any way.
    """

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        lags: Optional[List[int]] = None,
        batch_size: Optional[int] = 32,
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
        self.batch_size = batch_size

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
            batch_size=self.batch_size,
        )


def LocalTabularPredictor(*args, **kwargs) -> Localizer:
    return Localizer(TabularEstimator(*args, **kwargs))
