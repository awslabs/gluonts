# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from typing import Any, Iterator, List, Optional, Tuple, Union
from collections.abc import Sequence

import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from pathlib import Path, PosixPath
import ujson as json

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

import sklearn


class TimeSeries:
    """Base data structure for time series

    Attributes:
        values: Time series values. This can be a vector in the case of univariate time series, or
            a matrix for multivariate time series, with shape (Time, dimension).
        labels: Timestep label. These indicate the presence of an anomaly at each time.
        item_id: Identifies the time series, usually with a string.
        predicted_scores: (Optional) Indicate the current estimate of an anomaly at each timestep.
        indicator: Inform on injections on the time series.
            For example, The value -1 indicates may be interesting to sample windows containing this timestep.
            Could be used as a probability to sample from.
    """

    values: np.ndarray
    labels: np.ndarray
    item_id: Any
    predicted_scores: Optional[np.ndarray] = None
    indicator: np.ndarray

    def __init__(
        self,
        values: np.ndarray,
        labels: Optional[np.ndarray] = None,
        predicted_scores: Optional[np.ndarray] = None,
        item_id: Any = None,
        indicator: Optional[np.ndarray] = None,
    ):

        self.values = np.asarray(values, np.float32)
        self.labels = (
            np.asarray(labels, np.float32)
            if labels is not None
            else np.array([None] * len(values), dtype=np.float32)
        )
        assert len(self.values) == len(self.labels)
        self.item_id = item_id
        self.predicted_scores = predicted_scores
        self.indicator = np.zeros_like(self.labels).astype(int) if indicator is None else indicator

    def __len__(self):
        return len(self.values)

    def copy(self) -> "TimeSeries":
        return TimeSeries(
            values=self.values.copy(),
            labels=custom_copy(self.labels),
            predicted_scores=custom_copy(self.predicted_scores),
            indicator=custom_copy(self.indicator),
            item_id=self.item_id,
        )

    def to_json(self, path: Union[PosixPath, str]):
        path = PosixPath(path).expanduser()
        with open(path, "w") as f:
            json.dump(dict(values=self.values.tolist(), labels=self.labels.tolist()), f)

    def __repr__(self):
        if self.values.ndim == 1:
            T, ts_channels = self.values.shape[0], 1
        elif self.values.ndim == 2:
            T, ts_channels = self.values.shape
        else:
            raise ValueError("values must be a vector or matrix.")
        return f"TimeSeries( item_id:{self.item_id!r}, channels:{ts_channels}, T:{T} \n values:{self.values!r} \n labels:{self.labels!r} \n predicted_scores:{self.predicted_scores!r})"

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key + 1, 1)

        ts = TimeSeries(
            values=self.values[key],
            labels=self.labels[key],
            predicted_scores=custom_slice(self.predicted_scores, key),
            indicator=custom_slice(self.indicator, key),
            item_id=None if self.item_id is None else f"slice of {self.item_id}",
        )
        return ts

    def append(self, ts: "TimeSeries"):
        ts_new = TimeSeries(
            values=np.concatenate([self.values, ts.values], axis=0),
            labels=np.concatenate([self.labels, ts.labels], axis=0),
            predicted_scores=custom_concat(self.predicted_scores, ts.predicted_scores, axis=0),
            indicator=custom_concat(self.indicator, ts.indicator, axis=0),
            item_id=f"({self.item_id},{ts.item_id})",
        )
        return ts_new

    @property
    def shape(self):
        if self.values.ndim == 1:
            T, ts_channels = self.values.shape[0], 1
        elif self.values.ndim == 2:
            T, ts_channels = self.values.shape
        else:
            raise ValueError("values must be a vector or matrix.")
        return T, ts_channels

    @property
    def nan_ts_values(self):
        return np.isnan(self.values).any()

    def plot(self, title: str = ""):
        T, ts_channels = self.shape
        fig, axs = plt.subplots(ts_channels, 1, figsize=(10, 2 * ts_channels))
        if ts_channels == 1:
            axs = np.array([axs])

        for i in range(ts_channels):
            values = self.values if ts_channels == 1 else self.values[:, i]
            axs[i].plot(values)
            ts_extrema = np.nanquantile(values, [0, 1])
            if ts_extrema[0] == ts_extrema[1]:
                ts_extrema[0] -= 1
                ts_extrema[1] += 1
            axs[i].fill_between(
                x=range(T),
                y1=np.ones(T) * ts_extrema[0],
                y2=np.where(self.labels == 1, ts_extrema[1], ts_extrema[0]),
                color="red",
                alpha=0.5,
                label="suspect window",
            )
            axs[i].fill_between(
                x=range(T),
                y1=np.ones(T) * ts_extrema[0],
                y2=np.where(self.indicator == 1, ts_extrema[1], ts_extrema[0]),
                color="blue",
                alpha=0.2,
                label="suspect window",
            )
            axs[i].set_ylim(ts_extrema[0], ts_extrema[1])
        fig.suptitle(title)
        return fig, axs


class TimeSeriesDataset(List[TimeSeries]):
    """Collection of Time Series."""

    def copy(self):
        return TimeSeriesDataset([ts.copy() for ts in self])

    def __repr__(self):
        return f"TimeSeriesDataset: {len(self)} TimeSeries"

    def save_json_files(self, path: Union[PosixPath, str], ts_filename_chars: int = 5):
        path = PosixPath(path).expanduser()
        assert path.is_dir()
        for i, ts in enumerate(tqdm(self)):
            item_id = str(ts.item_id) if ts.item_id is not None else str(i).zfill(ts_filename_chars)
            ts_fn = path / f"{item_id}.json"
            ts.to_json(ts_fn)

    @property
    def nan_ts_values(self):
        return any([ts.nan_ts_values for ts in self])

    @property
    def shape(self):
        return [ts.shape for ts in self]

    def __getitem__(self, key):
        out = super().__getitem__(key)
        if isinstance(out, TimeSeries):
            return out
        elif isinstance(out, List):
            return TimeSeriesDataset(out)
        else:
            raise NotImplementedError()


def custom_copy(array: Optional[np.ndarray]):
    if array is None:
        return None
    return array.copy()


def custom_slice(array: Optional[np.ndarray], key):
    if array is None:
        return None
    return array[key]


def custom_concat(array1: Optional[np.ndarray], array2: Optional[np.ndarray], *args, **kwargs):
    if (array1 is None) and (array2 is None):
        return None
    return np.concatenate([array1, array2], *args, **kwargs)


##### Functions for TimeSeries and TimeSeriesDataset  #####


def split_train_val_test(
    data: TimeSeriesDataset,
    val_portion: float = 0.2,
    test_portion: float = 0.2,
    split_method: str = ["random_series", "sequential", "past_future", "past_future_with_warmup"][
        0
    ],
    split_warmup_length: Optional[int] = None,
    verbose: bool = True,
    *args,
    **kwargs,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """Split a TimeSeries dataset into three subsets (train, validation, test) with the requested proportions.

    Args:
        data: Dataset to be split.
        val_portion: Proportion of the dataset to include in the validation split.
            A value between 0.0 and 1.0
        test_portion: Proportion of the dataset to include in the test split.
            A value between 0.0 and 1.0
        split_method: Specifies the method used for splitting the Time Series.
            'random_series' : Randomly picks whole series and place them into a subset.
            'sequential' : Sequentially assign each time series to the subsets
                by considering the length of each series and the total length along all series.
            'past_future' : Each series is subdivided in (tran,val,test),
                the segments are assigned to the each subset correspondingly.
        verbose: If true, details of the split are printed.
    """

    train_portion = 1 - val_portion - test_portion
    assert (train_portion + val_portion + test_portion) == 1

    train_set = []
    val_set = []
    test_set = []

    if split_method == "random_series":
        train_set, test_set = sklearn.model_selection.train_test_split(data, test_size=test_portion)
        train_set, val_set = sklearn.model_selection.train_test_split(
            train_set, test_size=val_portion / (1 - test_portion)
        )

    elif split_method == "sequential":
        # this works best with many time series
        list_lengths = [len(ts.values) for ts in data]
        number_time_steps_total = sum(list_lengths)

        numb_time_step_train = (1 - val_portion - test_portion) * number_time_steps_total
        current_train_timestep = 0
        current_ts = 0

        while current_train_timestep < numb_time_step_train:
            train_set.append(data[current_ts])
            current_train_timestep += len(data[current_ts].values)
            current_ts += 1
            if len(data) - current_ts <= 2:
                break

        current_val_timestep = 0
        while current_val_timestep < val_portion * number_time_steps_total:
            val_set.append(data[current_ts])
            current_val_timestep += len(data[current_ts].values)
            current_ts += 1

            if len(data) - current_ts <= 1:
                break

        while current_ts < len(data):
            test_set.append(data[current_ts])
            current_ts += 1

        if verbose:
            print(len(train_set), len(val_set), len(test_set))
            print(
                current_train_timestep,
                current_val_timestep,
                number_time_steps_total - current_train_timestep - current_val_timestep,
            )

    elif split_method in ["past_future", "past_future_with_warmup"]:

        if split_method == "past_future_with_warmup":
            assert (
                split_warmup_length is not None
            ), "'split_warmup_length' must be given when using split_method='past_future_with_warmup'."
            assert split_warmup_length >= 0, "split_warmup_length must be a non-negative integer."

        for ts in data:
            # identify indices for splitting the series
            indices_split = len(ts.values) * np.array([train_portion, train_portion + val_portion])
            indices_split = np.floor(indices_split).astype(int)
            # print("new")
            # print(indices_split, train_portion,train_portion+val_portion)

            if split_method == "past_future_with_warmup":
                l_splits = [
                    (0, indices_split[0]),
                    (np.max([0, indices_split[0] - split_warmup_length]), indices_split[1]),
                    (np.max([0, indices_split[1] - split_warmup_length]), len(ts.values)),
                ]
                indices_split = l_splits.copy()

            # split the series
            ts_train, ts_val, ts_test = ts_split(ts, indices_split)
            # print(indices_split)
            # print(len(ts), len(ts_train), len(ts_val), len(ts_test), split_warmup_length)

            # if split_method == 'past_future_with_warmup':
            # append warmup from the preceding segment
            # assert ts_val.shape[0]>=split_warmup_length, 'Length of TimeSeries in the validation split must be >= "split_warmup_length"'
            # ts_test = ts_val[-split_warmup_length:].append( ts_test )
            #     assert ts_train.shape[0]>=split_warmup_length, 'Length of TimeSeries in the training split must be >= "split_warmup_length"'
            #     ts_val = ts_train[-split_warmup_length:].append( ts_val )

            # print(len(ts), len(ts_train), len(ts_val), len(ts_test), split_warmup_length)
            # Gather all time series splits
            train_set.append(ts_train)
            val_set.append(ts_val)
            test_set.append(ts_test)
    else:
        raise NotImplementedError(f"split_method={split_method} is not supported.")

    train_set = TimeSeriesDataset(train_set)
    val_set = TimeSeriesDataset(val_set)
    test_set = TimeSeriesDataset(test_set)

    return train_set, val_set, test_set


def ts_random_crop(
    ts: TimeSeries,
    length: int,
    num_crops: int = 1,
) -> TimeSeriesDataset:

    T = len(ts.values)

    if T < length:
        return []
    idx_end = np.random.randint(low=length, high=T, size=num_crops)

    out = [
        TimeSeries(
            values=_slice_pad_left(v=ts.values, end=idx_end[i], size=length, pad_value=np.nan),
            labels=_slice_pad_left(v=ts.labels, end=idx_end[i], size=length, pad_value=np.nan),
        )
        for i in range(num_crops)
    ]

    return out


def _slice_pad_left(v, end, size, pad_value):
    start = max(end - size, 0)
    v_slice = v[start:end]
    if len(v_slice) == size:
        return v_slice
    diff = size - len(v_slice)
    result = np.concatenate([[pad_value] * diff, v_slice])
    assert len(result) == size
    return result


def ts_rolling_window(
    ts: TimeSeries,
    window_length: int,
    stride: int = 1,
) -> TimeSeriesDataset:

    values_windows = _rolling_window(ts.values, window_length=window_length, stride=stride)
    labels_windows = _rolling_window(ts.labels, window_length=window_length, stride=stride)
    out = []
    n_windows = len(values_windows)
    for i in range(n_windows):
        out.append(TimeSeries(values=values_windows[i], labels=labels_windows[i]))

    return out


def _rolling_window(
    ts_array: np.ndarray,
    window_length: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Return a view to rolling windows of a time-series.
    """
    assert len(ts_array) >= window_length

    shape = (((len(ts_array) - window_length) // stride) + 1, window_length)
    strides = (ts_array.strides[0] * stride, ts_array.strides[0])
    return np.lib.stride_tricks.as_strided(ts_array, shape=shape, strides=strides)


def ts_split(
    ts: TimeSeries,
    indices_or_sections,
) -> TimeSeriesDataset:
    """Split a TimeSeries into multiple sub-TimeSeries.

    Args:
        ts : Time series to split
        indices_or_sections : Specify how to split,
            same logic as in np.split()
    """

    if not isinstance(indices_or_sections[0], Sequence):
        values_split = np.split(ts.values, indices_or_sections)
        labels_split = np.split(ts.labels, indices_or_sections)
        predicted_scores_split = (
            np.split(ts.predicted_scores, indices_or_sections) if ts.predicted_scores else None
        )
    else:
        values_split = _overlapping_split(ts.values, indices_or_sections)
        labels_split = _overlapping_split(ts.labels, indices_or_sections)
        predicted_scores_split = (
            _overlapping_split(ts.predicted_scores, indices_or_sections)
            if ts.predicted_scores
            else None
        )

    out = TimeSeriesDataset(
        [
            TimeSeries(
                values=values_split[i],
                labels=labels_split[i],
                item_id=f"{ts.item_id}_{i}" if ts.item_id else None,
                predicted_scores=predicted_scores_split[i] if ts.predicted_scores else None,
            )
            for i in range(len(values_split))
        ]
    )
    return out


def _overlapping_split(array, list_indices):
    l_splits = []
    for i in range(len(list_indices)):
        l_splits.append(array[list_indices[i][0] : list_indices[i][1]])
    return l_splits


def ts_to_array(
    ts: TimeSeriesDataset,
) -> np.array:
    """TimeSeries to numpy array

    Obtains a numpy array by stacking values and labels from a TimeSeries.
    The output dimension is (T,ts_channels+1)
    """
    out = np.concatenate((ts.values.reshape(ts.shape), ts.labels.reshape(ts.shape)), axis=1)
    return out
