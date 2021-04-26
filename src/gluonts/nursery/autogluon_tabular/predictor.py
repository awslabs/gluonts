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

from typing import Callable, Dict, List, Optional, Iterator, Iterable, Tuple

import numpy as np
import pandas as pd
import shutil
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pathlib import Path
from autogluon.tabular import TabularPredictor as AutogluonTabularPredictor

from gluonts.core.serde import dump_json, load_json
from gluonts.dataset.util import to_pandas
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import Predictor
from gluonts.time_feature import TimeFeature


def no_scaling(series: pd.Series):
    return series, 1.0


def mean_abs_scaling(series: pd.Series, minimum_scale=1e-6):
    """Scales a Series by the mean of its absolute value. Returns the scaled Series
    and the scale itself.
    """
    scale = max(minimum_scale, series.abs().mean())
    return series / scale, scale


def get_features_dataframe(
    series: pd.Series,
    time_features: List[TimeFeature],
    lag_indices: List[int],
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
    time_features
        List of time features to be included in the data frame.
    lag_indices
        List of indices of lagged observations to be included as features.
    past_data
        Prior data, to be used to compute lagged observations.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the features. This has the same index as `series`.
    """
    # TODO check if anything can be optimized here

    assert past_data is None or series.index.freq == past_data.index.freq
    assert past_data is None or series.index[0] > past_data.index[-1]

    cal = calendar()
    holidays = cal.holidays(start=series.index.min(), end=series.index.max())
    time_feature_columns = {
        feature.__class__.__name__: feature(series.index)
        for feature in time_features
    }

    all_data = (
        series
        if past_data is None
        else past_data.append(series).asfreq(series.index.freq)
    )
    lag_columns = {
        f"lag_{idx}": all_data.shift(idx)[series.index].values
        for idx in lag_indices
    }

    columns = {**time_feature_columns, **lag_columns, "target": series.values}

    return pd.DataFrame(columns, index=series.index)


class TabularPredictor(Predictor):
    def __init__(
        self,
        ag_model,
        freq: str,
        prediction_length: int,
        time_features: List[TimeFeature],
        lag_indices: List[int],
        scaling: Callable[[pd.Series], Tuple[pd.Series, float]],
        batch_size: Optional[int] = 32,
        dtype=np.float32,
    ) -> None:
        super().__init__(prediction_length=prediction_length, freq=freq)
        assert all(l >= 1 for l in lag_indices)

        self.ag_model = ag_model
        self.time_features = time_features
        self.lag_indices = lag_indices
        self.scaling = scaling
        self.batch_size = batch_size
        self.dtype = dtype

    @property
    def auto_regression(self) -> bool:
        return (
            False
            if not self.lag_indices
            else self.prediction_length > min(self.lag_indices)
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
            series, scale = self.scaling(to_pandas(entry))

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
                    forecast_series,
                    time_features=self.time_features,
                    lag_indices=self.lag_indices,
                    past_data=series,
                )
                full_series[forecast_series.index] = self.ag_model.predict(df)

            else:  # predict step by step
                for idx in forecast_series.index:
                    df = get_features_dataframe(
                        forecast_series[idx:idx],
                        time_features=self.time_features,
                        lag_indices=self.lag_indices,
                        past_data=full_series[:idx][:-1],
                    )
                    full_series[idx] = self.ag_model.predict(df).item()

            yield self._to_forecast(
                scale * full_series[forecast_index].values.astype(self.dtype),
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
        item_ids = []
        scales = []
        forecast_start_timestamps = []
        dfs = []

        for entry in dataset:
            item_ids.append(entry.get(FieldName.ITEM_ID, None))
            series, scale = self.scaling(to_pandas(entry))
            scales.append(scale)
            forecast_start = series.index[-1] + series.index.freq
            forecast_start_timestamps.append(forecast_start)
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
                    forecast_series,
                    time_features=self.time_features,
                    lag_indices=self.lag_indices,
                    past_data=series,
                )
            )

        df = pd.concat(dfs)
        output = self.ag_model.predict(df)

        for arr, scale, forecast_start, item_id in zip(
            np.split(output, len(dfs)),
            scales,
            forecast_start_timestamps,
            item_ids,
        ):
            yield self._to_forecast(
                scale * arr.values,
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
        batch_ids = []
        batch_scales = []
        batch_series = []

        for entry in dataset:
            batch_ids.append(entry.get(FieldName.ITEM_ID, None))
            series, scale = self.scaling(to_pandas(entry))
            batch_scales.append(scale)
            batch_series.append(series)

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
                        time_features=self.time_features,
                        lag_indices=self.lag_indices,
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

        for arr, scale, forecast_index, item_id in zip(
            output, batch_scales, batch_forecast_indices, batch_ids
        ):
            yield self._to_forecast(
                scale * arr,
                forecast_index[0],
                item_id=item_id,
            )

    def _predict_batch(
        self, dataset: Iterable[Dict], batch_size: int, **kwargs
    ) -> Iterator[SampleForecast]:
        for batch in batcher(dataset, batch_size):
            yield from (
                self._predict_batch_autoreg(batch, **kwargs)
                if self.auto_regression
                else self._predict_batch_one_shot(batch, **kwargs)
            )

    def predict(
        self,
        dataset: Iterable[Dict],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Iterator[SampleForecast]:
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size > 0
        if batch_size is None or batch_size == 1:
            return self._predict_serial(dataset, **kwargs)
        else:
            return self._predict_batch(
                dataset, batch_size=batch_size, **kwargs
            )

    def serialize(self, path: Path) -> None:
        # call Predictor.serialize() in order to serialize the class name

        super().serialize(path)

        # serialize self.ag_model
        # move autogluon model to where we want to do the serialization
        ag_path = self.ag_model.path
        shutil.move(ag_path, path)
        ag_path = Path(ag_path)
        print(f"Autogluon files moved from {ag_path} to {path}.")
        # reset the path stored in tabular model.
        AutogluonTabularPredictor.load(path / Path(ag_path.name))
        # serialize all remaining constructor parameters
        with (path / "parameters.json").open("w") as fp:
            parameters = dict(
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                freq=self.freq,
                dtype=self.dtype,
                time_features=self.time_features,
                lag_indices=self.lag_indices,
                ag_path=path / Path(ag_path.name),
            )
            print(dump_json(parameters), file=fp)

    @classmethod
    def deserialize(
        cls,
        path: Path,
        # TODO this is temporary, we should make the callable object serializable in the first place
        scaling: Callable[
            [pd.Series], Tuple[pd.Series, float]
        ] = mean_abs_scaling,
        **kwargs,
    ) -> "Predictor":
        # deserialize constructor parameters
        with (path / "parameters.json").open("r") as fp:
            parameters = load_json(fp.read())
        loaded_ag_path = parameters["ag_path"]
        del parameters["ag_path"]
        # load tabular model
        ag_model = AutogluonTabularPredictor.load(loaded_ag_path)

        return TabularPredictor(
            ag_model=ag_model, scaling=scaling, **parameters
        )
