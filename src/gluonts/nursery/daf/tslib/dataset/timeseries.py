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


from __future__ import annotations

from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Sequence,
    Optional,
    Callable,
    overload,
)
from operator import eq
from collections import defaultdict
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from numpy import ndarray
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as pt


def _dict_equal(dict1: Dict, dict2: Dict, cmp_func: Callable = eq) -> bool:
    if dict1.keys() == dict2.keys():
        return all(cmp_func(dict1[k], dict2[k]) for k in dict1.keys())
    return False


def _pad_array_along_axis(
    array: ndarray, pad: Tuple[int, int], axis: int = 0, constant_values=np.nan
) -> ndarray:
    full_pad = [(0, 0)] * array.ndim
    full_pad[axis] = pad
    return np.pad(
        array, full_pad, mode="constant", constant_values=constant_values
    )


class TimeSeriesInstant(object):
    def __init__(
        self,
        target: ndarray,
        timestamp: pd.Timestamp,
        series_name: Optional[str],
        target_names: List[str],
    ):
        if target.ndim == 1:
            self.target = target
        else:
            raise ValueError("target has to be a 1-d array")
        self.timestamp = timestamp
        self.series_name = series_name
        self.target_names = target_names

        self._features = dict()
        self._categorical_features = set()
        self._numerical_features = set()

    def add_features(self, name: str, data: ndarray, data_type: str) -> None:
        valid_data_type = ["cat", "num", "categorical", "numerical"]
        data_type = data_type.lower()
        if data_type not in valid_data_type:
            raise ValueError(f"data_type should be one of {valid_data_type}")
        numerical = data_type in ["num", "numerical"]

        valid_ndim = 1 if numerical else 0
        if data.ndim != valid_ndim:
            raise ValueError(
                f"{feature_type} and {data_type} type feature requires {valid_ndim}-d data"
            )

        self._features[name] = data
        if numerical:
            self._numerical_features.add(name)
        else:
            self._categorical_features.add(name)

    @property
    def categorical_features(self) -> Dict:
        return {
            name: self._features[name] for name in self._categorical_features
        }

    @property
    def numerical_features(self) -> Dict:
        return {
            name: self._features[name] for name in self._numerical_features
        }

    @property
    def d_data(self) -> int:
        return self.target.shape[1]

    def __eq__(self, other) -> bool:
        return all(
            [
                isinstance(other, TimeSeriesInstant),
                np.array_equal(self.target, other.target),
                np.array_equal(self.timestamp, other.timestamp),
                self.series_name == other.series_name,
                np.array_equal(self.target_names, other.target_names),
                _dict_equal(
                    self.categorical_features,
                    other.categorical_features,
                    np.array_equal,
                ),
                _dict_equal(
                    self.numerical_features,
                    other.numerical_features,
                    np.array_equal,
                ),
            ]
        )

    def __repr__(self):
        string = f"time = {self.timestamp:%Y-%m-%d %H:%M:%S}\n"
        string += f"data = {self.target}\n"
        for name, val in self.categorical_features.items():
            string += f"{name} = {val}\n"
        for name, val in self.numerical_features.items():
            string += f"{name} = {val}\n"
        return string


class TimeSeries(Sequence[TimeSeriesInstant]):
    def __init__(
        self,
        target: ndarray,
        start_time: Optional[pd.Timestamp] = None,
        time_unit: Optional[str] = None,
        time_index: Optional[pd.DatetimeIndex] = None,
        series_name: Optional[str] = None,
        target_names: Optional[Union[str, List[str]]] = None,
    ):
        if target.ndim == 1:
            self.target = target.reshape(-1, 1)
        else:
            self.target = target
        if time_index is None:
            assert not (
                (start_time is None) or (time_unit is None)
            ), "if `time_index` is not given, both `start_time` and `time_unit` must be specified"
            self.time_index = pd.date_range(
                start=start_time, periods=len(self), freq=time_unit
            )
            self.time_unit = time_unit
        else:
            assert len(time_index) == len(self.target)
            self.time_index = time_index
            self.time_unit = time_unit or pd.infer_freq(time_index)
        self.series_name = series_name or "Unknown"
        if target_names is None:
            self.target_names = [f"d_{i+1}" for i in range(self.d_data)]
        elif isinstance(target_names, str):
            if self.d_data > 1:
                self.target_names = [
                    f"{target_names}_{i+1}" for i in range(self.d_data)
                ]
            else:
                self.target_names = [target_names]
        else:
            err_msg = f"expected {self.d_data}, given {len(target_names)}."
            if len(target_names) > self.d_data:
                raise ValueError(f"Too many target names are given, {err_msg}")
            if len(target_names) < self.d_data:
                raise ValueError(
                    f"Not enough target names are given, {err_msg}"
                )
            self.target_names = target_names

        self._features = dict()
        self._static_features = set()
        self._revealed_features = set()
        self._observed_features = set()
        self._categorical_features = set()
        self._numerical_features = set()

    def add_features(
        self, name: str, data: ndarray, data_type: str, feature_type: str
    ) -> None:
        valid_data_type = ["cat", "num", "categorical", "numerical"]
        valid_feature_type = ["static", "revealed", "observed"]
        data_type = data_type.lower()
        feature_type = feature_type.lower()
        if data_type not in valid_data_type:
            raise ValueError(f"data_type should be one of {valid_data_type}")
        if feature_type not in valid_feature_type:
            raise ValueError(
                f"feature_type should be one of {valid_feature_type}"
            )

        numerical = data_type in ["num", "numerical"]
        static = feature_type == "static"
        revealed = feature_type == "revealed"

        valid_ndim = 1 if numerical else 0
        if not static:
            valid_ndim += 1
        if data.ndim != valid_ndim:
            raise ValueError(
                f"{feature_type} and {data_type} type feature requires {valid_ndim}-d data"
            )

        self._features[name] = data
        if static:
            self._static_features.add(name)
        elif revealed:
            self._revealed_features.add(name)
        else:
            self._observed_features.add(name)

        if numerical:
            self._numerical_features.add(name)
        else:
            self._categorical_features.add(name)

    @property
    def static_categorical_features(self) -> Dict:
        return {
            name: self._features[name]
            for name in self._static_features & self._categorical_features
        }

    @property
    def static_numerical_features(self) -> Dict:
        return {
            name: self._features[name]
            for name in self._static_features & self._numerical_features
        }

    @property
    def revealed_categorical_features(self) -> Dict:
        return {
            name: self._features[name]
            for name in self._revealed_features & self._categorical_features
        }

    @property
    def revealed_numerical_features(self) -> Dict:
        return {
            name: self._features[name]
            for name in self._revealed_features & self._numerical_features
        }

    @property
    def observed_categorical_features(self) -> Dict:
        return {
            name: self._features[name]
            for name in self._observed_features & self._categorical_features
        }

    @property
    def observed_numerical_features(self) -> Dict:
        return {
            name: self._features[name]
            for name in self._observed_features & self._numerical_features
        }

    @property
    def _dynamic_features(self) -> set:
        return self._observed_features | self._revealed_features

    @property
    def dynamic_categorical_features(self) -> Dict:
        return {
            name: self._features[name]
            for name in self._dynamic_features & self._categorical_features
        }

    @property
    def dynamic_numerical_features(self) -> Dict:
        return {
            name: self._features[name]
            for name in self._dynamic_features & self._numerical_features
        }

    def __eq__(self, other):
        return all(
            [
                isinstance(other, TimeSeries),
                np.array_equal(self.target, other.target),
                np.array_equal(self.time_index, other.time_index),
                self.series_name == other.series_name,
                np.array_equal(self.target_names, other.target_names),
                _dict_equal(
                    self.static_categorical_features,
                    other.static_categorical_features,
                    np.array_equal,
                ),
                _dict_equal(
                    self.static_numerical_features,
                    other.static_numerical_features,
                    np.array_equal,
                ),
                _dict_equal(
                    self.revealed_categorical_features,
                    other.revealed_categorical_features,
                    np.array_equal,
                ),
                _dict_equal(
                    self.revealed_numerical_features,
                    other.revealed_numerical_features,
                    np.array_equal,
                ),
                _dict_equal(
                    self.observed_categorical_features,
                    other.observed_categorical_features,
                    np.array_equal,
                ),
                _dict_equal(
                    self.observed_numerical_features,
                    other.observed_numerical_features,
                    np.array_equal,
                ),
            ]
        )

    def __len__(self):
        return len(self.target)

    @overload
    def index_by_timestamp(self, index: pd.Timestamp) -> int:
        ...

    @overload
    def index_by_timestamp(self, index: List[pd.Timestamp]) -> List[int]:
        ...

    def index_by_timestamp(self, index):
        return pd.Series(
            index=self.time_index, data=np.arange(len(self.time_index))
        ).loc[index]

    @overload
    def __getitem__(self, index: int) -> TimeSeriesInstant:
        ...

    @overload
    def __getitem__(self, index: pd.Timestamp) -> TimeSeriesInstant:
        ...

    @overload
    def __getitem__(self, index: slice) -> TimeSeries:
        ...

    @overload
    def __getitem__(self, index: List[int]) -> TimeSeries:
        ...

    @overload
    def __getitem__(self, index: List[pd.Timestamp]) -> TimeSeries:
        ...

    def __getitem__(self, index):
        if isinstance(index, pd.Timestamp) or (
            isinstance(index, List) and isinstance(index[0], pd.Timestamp)
        ):
            index = self.index_by_timestamp(index)

        left_pad = right_pad = None
        if isinstance(index, slice):
            if (index.stop is not None) and (index.stop > 0):
                stop = index.stop
                if stop > len(self):
                    right_pad = stop - len(self)
                    stop = None
                start = index.start
                if (index.start is not None) and (index.start < 0):
                    left_pad = -index.start
                    start = None
                index = slice(start, stop)
        target = self.target[index]
        time_index = self.time_index[index]

        if isinstance(index, int):
            obj = TimeSeriesInstant(
                target,
                time_index,
                series_name=self.series_name,
                target_names=self.target_names,
            )
            for name in self._categorical_features:
                if name in self._static_features:
                    data = self._features[name]
                else:
                    data = self._features[name][index]
                obj.add_features(name, data, data_type="cat")
            for name in self._numerical_features:
                if name in self._static_features:
                    data = self._features[name]
                else:
                    data = self._features[name][index]
                obj.add_features(name, data, data_type="num")
            return obj
        else:
            if left_pad is not None:
                if self.time_unit is not None:
                    td = pd.Timedelta(1, unit=self.time_unit).to_numpy()
                else:
                    if len(time_index) < 2:
                        raise IndexError(
                            "Cannot pad time index with a single timestamp"
                        )
                    td = time_index[1] - time_index[0]
                time_index = np.r_[
                    np.arange(
                        time_index[0] - td * left_pad, time_index[0], td
                    ),
                    time_index,
                ]
                target = _pad_array_along_axis(
                    target, (left_pad, 0), constant_values=np.nan
                )
            if right_pad is not None:
                if self.time_unit is not None:
                    td = pd.Timedelta(1, unit=self.time_unit).to_numpy()
                else:
                    if len(time_index) < 2:
                        raise IndexError(
                            "Cannot pad time index with a single timestamp"
                        )
                    td = time_index[-1] - time_index[-2]
                time_index = np.r_[
                    time_index,
                    np.arange(
                        time_index[-1] + td,
                        time_index[-1] + td * (right_pad + 1),
                        td,
                    ),
                ]
                target = _pad_array_along_axis(
                    target, (0, right_pad), constant_values=np.nan
                )
            obj = TimeSeries(
                target,
                time_index=time_index,
                series_name=self.series_name,
                target_names=self.target_names,
            )
            for name in self._static_features:
                data_type = (
                    "cat" if name in self._categorical_features else "num"
                )
                data = self._features[name]
                obj.add_features(
                    name, data, data_type=data_type, feature_type="static"
                )
            for name in self._revealed_features | self._observed_features:
                data_type = (
                    "cat" if name in self._categorical_features else "num"
                )
                feature_type = (
                    "revealed"
                    if name in self._revealed_features
                    else "observed"
                )
                val = -1 if data_type == "cat" else np.nan
                data = self._features[name][index]
                if left_pad is not None:
                    data = _pad_array_along_axis(
                        data, (left_pad, 0), constant_values=val
                    )
                if right_pad is not None:
                    data = _pad_array_along_axis(
                        data, (0, right_pad), constant_values=val
                    )
                obj.add_features(
                    name, data, data_type=data_type, feature_type=feature_type
                )
            return obj

    @property
    def d_data(self) -> int:
        return self.target.shape[1]

    @classmethod
    def from_pandas_series(
        cls, series: pd.Series, target_name: Optional[str] = None
    ):
        return cls(
            series.values,
            time_index=series.index,
            series_name=series.name,
            target_names=target_name,
        )

    def to_pandas_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self.target, columns=self.target_names, index=self.time_index
        )
        for name in self._static_features:
            data = self._features[name]
            if data.ndim == 0 or (data.ndim == 1 and data.shape[0] == 1):
                df[name] = np.tile(data, len(self))
            else:
                for i in range(data.shape[0]):
                    df[f"{name}_{i}"] = np.tile(data[i], len(self))
        for name in self._revealed_features | self._observed_features:
            data = self._features[name]
            if data.ndim == 1:
                df[name] = data
            elif data.ndim == 2 and data.shape[1] == 1:
                df[name] = data[:, 0]
            else:
                for i in range(data.shape[0]):
                    df[f"{name}_{i}"] = data[:, i]
        return df

    def to_pandas_series(self) -> pd.Series:
        df = self.to_pandas_dataframe()
        series = df[df.columns[0]]
        series.name = "1D-data"
        return series

    def __repr__(self) -> str:
        string = f"Time series {self.series_name} of size {len(self)}x{self.d_data}\n"
        string += f"static_features: {self._static_features}\n"
        string += f"revealed_features: {self._revealed_features}\n"
        string += f"observed_features: {self._observed_features}\n"
        string += f"categorical_features: {self._categorical_features}\n"
        string += f"numerical_features: {self._numerical_features}\n"
        return string


class TimeSeriesCorpus(Sequence[TimeSeries]):
    def __init__(
        self,
        instances: List[TimeSeries],
        scopes: Optional[List[Tuple[int, slice]]] = None,
        categorical_encoders: Optional[defaultdict] = None,
        numerical_encoders: Optional[defaultdict] = None,
        add_series_id: bool = False,
    ) -> None:
        if categorical_encoders is None and numerical_encoders is None:
            self.categorical_encoders = defaultdict(LabelEncoder)
            self.numerical_scalers = defaultdict(StandardScaler)
            self.instances = self._check_consistency(instances)
        else:
            self.categorical_encoders = categorical_encoders
            self.numerical_scalers = numerical_encoders
            self.instances = instances

        if scopes is None:
            self.scopes = [
                (ts_id, slice(len(ts)))
                for ts_id, ts in enumerate(self.instances)
            ]
        else:
            self.scopes = scopes
        self.add_series_id = add_series_id

    def _check_consistency(
        self, instances: List[TimeSeries]
    ) -> List[TimeSeries]:
        def _consistent(ts1: TimeSeries, ts2: TimeSeries) -> bool:
            return all(
                [
                    np.array_equal(ts1.target_names, ts2.target_names),
                    ts1._static_features == ts2._static_features,
                    ts1._revealed_features == ts2._revealed_features,
                    ts1._observed_features == ts2._observed_features,
                    ts1._categorical_features == ts2._categorical_features,
                    ts1._numerical_features == ts2._numerical_features,
                ]
            )

        cats = defaultdict(list)
        nums = defaultdict(list)
        ref = instances[0]
        for i, ts in enumerate(instances):
            if not _consistent(ref, ts):
                raise ValueError("Given time series are not consistent")
            for name in ts._categorical_features:
                cats[name].append(ts._features[name])
            for name in ts._numerical_features & ts._static_features:
                nums[name].append(ts._features[name])
        for name, vals in cats.items():
            data = np.stack(vals, axis=0).flatten()
            self.categorical_encoders[name].fit(data)
        for name, vals in nums.items():
            data = np.stack(vals, axis=0)
            self.numerical_scalers[name].fit(data)
        return instances

    def __len__(self) -> int:
        return len(self.scopes)

    def __getitem__(self, index: int) -> TimeSeries:
        ts_id, scope = self.scopes[index]
        ts = self.instances[ts_id][scope]
        for name in ts._categorical_features:
            data = ts._features[name]
            shape = data.shape
            data = self.categorical_encoders[name].transform(data.flatten())
            ts._features[name] = data.reshape(shape)
        for name in ts._numerical_features & ts._static_features:
            data = ts._features[name]
            shape = data.shape
            if data.ndim == 1:
                data = data[None, :]
            data = self.numerical_scalers[name].transform(data)
            ts._features[name] = data.reshape(shape)
        if self.add_series_id:
            ts.add_features(
                "id", np.array(ts_id), data_type="cat", feature_type="static"
            )
        return ts

    def __getattr__(self, key: str):
        value = getattr(
            super(TimeSeriesCorpus, self).__getattribute__("instances")[0], key
        )
        if key in ["_categorical_features", "_static_features"]:
            if self.add_series_id:
                value = value | {"id"}
        return value

    @property
    def cardinalities(self) -> Dict:
        cardi = {
            name: len(enc.classes_)
            for name, enc in self.categorical_encoders.items()
        }
        if self.add_series_id:
            cardi["id"] = len(self)
        return cardi

    def split_by_timestamp(
        self,
        split_timestamp: pd.Timestamp,
        horizon: Optional[int] = None,
        context: Optional[int] = None,
        n_rolls: int = 1,
        shift: Optional[int] = None,
        raise_error: bool = True,
    ) -> Tuple[TimeSeriesCorpus, TimeSeriesCorpus]:
        shift = shift or horizon
        train_scopes = []
        test_scopes = []
        for ts_id, scope in self.scopes:
            ts = self.instances[ts_id]
            start = scope.start or 0
            stop = scope.stop or len(ts)
            split_index = ts.index_by_timestamp(split_timestamp)
            test_start = start if context is None else split_index - context
            if horizon is None:
                test_stop = stop
            else:
                test_stop = split_index + shift * (n_rolls - 1) + horizon
            if not ((start <= test_start) and (test_stop <= stop)):
                err_msg = (
                    f"Scope {ts_id}, requested {test_start}:{test_stop} is not "
                    f"contained by the current scope {start}:{stop}."
                )
                if raise_error:
                    raise ValueError(err_msg)
                warnings.warn(f"{err_msg} No scope added to test split.")
                train_scopes.append((ts_id, slice(start, stop)))
            else:
                test_scopes.append((ts_id, slice(test_start, test_stop)))
                train_scopes.append((ts_id, slice(start, split_index)))
        train_corpus = TimeSeriesCorpus(
            self.instances,
            train_scopes,
            categorical_encoders=self.categorical_encoders,
            numerical_encoders=self.numerical_scalers,
            add_series_id=self.add_series_id,
        )
        test_corpus = TimeSeriesCorpus(
            self.instances,
            test_scopes,
            categorical_encoders=self.categorical_encoders,
            numerical_encoders=self.numerical_scalers,
            add_series_id=self.add_series_id,
        )
        return train_corpus, test_corpus

    def split_from_end(
        self,
        horizon: int,
        context: Optional[int] = None,
        n_rolls: int = 1,
        shift: Optional[int] = None,
        raise_error: bool = True,
    ) -> Tuple[TimeSeriesCorpus, TimeSeriesCorpus]:
        shift = shift or horizon
        train_scopes = []
        test_scopes = []
        for ts_id, scope in self.scopes:
            new_start = scope.stop - shift * (n_rolls - 1) - horizon
            start = scope.start or 0
            test_start = (
                scope.start if context is None else new_start - context
            )
            if test_start < start:
                err_msg = (
                    f"Scope {ts_id}, requested {test_start}:{scope.stop} is not "
                    f"contained by the current scope {start}:{scope.stop}."
                )
                if raise_error:
                    raise ValueError(err_msg)
                warnings.warn(f"{err_msg} No scope added to test split.")
                train_scopes.append((ts_id, slice(start, scope.stop)))
            else:
                test_scopes.append((ts_id, slice(test_start, scope.stop)))
                train_scopes.append((ts_id, slice(scope.start, new_start)))
        train_corpus = TimeSeriesCorpus(
            self.instances,
            train_scopes,
            categorical_encoders=self.categorical_encoders,
            numerical_encoders=self.numerical_scalers,
            add_series_id=self.add_series_id,
        )
        test_corpus = TimeSeriesCorpus(
            self.instances,
            test_scopes,
            categorical_encoders=self.categorical_encoders,
            numerical_encoders=self.numerical_scalers,
            add_series_id=self.add_series_id,
        )
        return train_corpus, test_corpus

    def split_from_end_by_blocks(
        self,
        horizon: int,
        context: Optional[int] = None,
        n_rolls: int = 1,
        n_blocks: int = 1,
        shift: Optional[int] = None,
        raise_error: bool = True,
    ) -> Tuple[TimeSeriesCorpus, TimeSeriesCorpus]:
        shift = shift or horizon
        train_scopes = []
        test_scopes = []
        # assert self.instances == self.scopes
        for ts_id, scope in self.scopes:
            block_size = len(self.instances[ts_id][scope]) // n_blocks
            ini = scope.start or 0
            for i in range(n_blocks):
                start = ini + i * block_size
                stop = start + block_size
                new_start = stop - shift * (n_rolls - 1) - horizon
                test_start = start if context is None else new_start - context
                if test_start < start:
                    err_msg = (
                        f"Scope {ts_id}, requested {test_start}:{stop} is not "
                        f"contained by the current block {start}:{top}."
                    )
                    if raise_error:
                        raise ValueError(err_msg)
                    warnings.warn(f"{err_msg} No scope added to test split.")
                    train_scopes.append((ts_id, slice(start, stop)))
                else:
                    test_scopes.append((ts_id, slice(test_start, stop)))
                    train_scopes.append((ts_id, slice(start, new_start)))
        train_corpus = TimeSeriesCorpus(
            self.instances,
            train_scopes,
            categorical_encoders=self.categorical_encoders,
            numerical_encoders=self.numerical_scalers,
            add_series_id=self.add_series_id,
        )
        test_corpus = TimeSeriesCorpus(
            self.instances,
            test_scopes,
            categorical_encoders=self.categorical_encoders,
            numerical_encoders=self.numerical_scalers,
            add_series_id=self.add_series_id,
        )
        return train_corpus, test_corpus


if __name__ == "__main__":
    instances = []
    for i in range(100):
        start_time = pd.Timestamp("2011-1-1")
        time_unit = "d"
        data = np.random.randn(100, 2)
        target_names = "data"
        t = TimeSeries(
            data,
            series_name=f"instance_{i+1}",
            start_time=start_time,
            time_unit=time_unit,
            target_names=target_names,
        )
        t.add_features("f1", np.random.randn(100, 1), "num", "revealed")
        t.add_features("f2", np.random.randn(100, 1), "num", "observed")
        t.add_features(
            "f3", np.random.randint(100, 200, (100,)), "cat", "revealed"
        )
        t.add_features(
            "f4", np.random.randint(1000, 1200, (100,)), "cat", "observed"
        )
        t.add_features("f5", np.array(i + 1), "cat", "static")
        t.add_features("f6", np.random.randn(2) + 12, "num", "static")
        instances.append(t)
    c = TimeSeriesCorpus(instances, add_series_id=True)

    t1, t2 = c.split_by_timestamp(pd.Timestamp("2011-04-01"), 4, 10)
    t1, t2 = c.split_from_end(4, 10)
    t1, t2 = c.split_from_end_by_blocks(4, 10, 4, 4)
