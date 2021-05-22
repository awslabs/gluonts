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

from functools import lru_cache
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.core.exception import GluonTSDateBoundsError
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName

from ._base import FlatMapTransformation
from .sampler import ContinuousTimePointSampler, InstanceSampler


def shift_timestamp(ts: pd.Timestamp, offset: int) -> pd.Timestamp:
    """
    Computes a shifted timestamp.

    Basic wrapping around pandas ``ts + offset`` with caching and exception
    handling.
    """
    return _shift_timestamp_helper(ts, ts.freq, offset)


@lru_cache(maxsize=10000)
def _shift_timestamp_helper(
    ts: pd.Timestamp, freq: str, offset: int
) -> pd.Timestamp:
    """
    We are using this helper function which explicitly uses the frequency as a
    parameter, because the frequency is not included in the hash of a time
    stamp.

    I.e.
      pd.Timestamp(x, freq='1D')  and pd.Timestamp(x, freq='1min')

    hash to the same value.
    """
    try:
        # this line looks innocent, but can create a date which is out of
        # bounds values over year 9999 raise a ValueError
        # values over 2262-04-11 raise a pandas OutOfBoundsDatetime
        return ts + offset * ts.freq
    except (ValueError, pd._libs.OutOfBoundsDatetime) as ex:
        raise GluonTSDateBoundsError(ex)


class InstanceSplitter(FlatMapTransformation):
    """
    Selects training instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.

    The target and each time_series_field is removed and instead two
    corresponding fields with prefix `past_` and `future_` are included. E.g.

    If the target array is one-dimensional, the resulting instance has shape
    (len_target). In the multi-dimensional case, the instance has shape (dim,
    len_target).

    target -> past_target and future_target

    The transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not.

    Convention: time axis is always the last axis.

    Parameters
    ----------

    target_field
        field containing the target
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        output field that will contain the time point where the forecast starts
    instance_sampler
        instance sampler that provides sampling indices given a time-series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    lead_time
        gap between the past and future windows (default: 0)
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout (default: True)
    time_series_fields
        fields that contains time-series, they are split in the same interval
        as the target (default: None)
    dummy_value
        Value to use for padding. (default: 0.0)
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: Optional[List[str]] = None,
        dummy_value: float = 0.0,
    ) -> None:

        assert future_length > 0

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = (
            time_series_fields if time_series_fields is not None else []
        )
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pl = self.future_length
        lt = self.lead_time
        slice_cols = self.ts_fields + [self.target_field]
        target = data[self.target_field]

        len_target = target.shape[-1]

        sampled_indices = self.instance_sampler(target)

        for i in sampled_indices:
            pad_length = max(self.past_length - i, 0)
            d = data.copy()
            for ts_field in slice_cols:
                if i > self.past_length:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.past_length : i]
                elif i < self.past_length:
                    pad_block = (
                        np.ones(
                            d[ts_field].shape[:-1] + (pad_length,),
                            dtype=d[ts_field].dtype,
                        )
                        * self.dummy_value
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                else:
                    past_piece = d[ts_field][..., :i]
                d[self._past(ts_field)] = past_piece
                d[self._future(ts_field)] = d[ts_field][
                    ..., i + lt : i + lt + pl
                ]
                del d[ts_field]
            pad_indicator = np.zeros(self.past_length, dtype=target.dtype)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            if self.output_NTC:
                for ts_field in slice_cols:
                    d[self._past(ts_field)] = d[
                        self._past(ts_field)
                    ].transpose()
                    d[self._future(ts_field)] = d[
                        self._future(ts_field)
                    ].transpose()

            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = shift_timestamp(
                d[self.start_field], i + lt
            )
            yield d


class CanonicalInstanceSplitter(FlatMapTransformation):
    """
    Selects instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.

    In training mode, the returned instances contain past_`target_field`
    as well as past_`time_series_fields`.

    In prediction mode, one can set `use_prediction_features` to get
    future_`time_series_fields`.

    If the target array is one-dimensional, the `target_field` in the resulting instance has shape
    (`instance_length`). In the multi-dimensional case, the instance has shape (`dim`, `instance_length`),
    where `dim` can also take a value of 1.

    In the case of insufficient number of time series values, the
    transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not, and the value is padded with
    `default_pad_value` with a default value 0.
    This is done only if `allow_target_padding` is `True`,
    and the length of `target` is smaller than `instance_length`.

    Parameters
    ----------
    target_field
        fields that contains time-series
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        field containing the forecast start date
    instance_sampler
        instance sampler that provides sampling indices given a time-series
    instance_length
        length of the target seen before making prediction
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout
    time_series_fields
        fields that contains time-series, they are split in the same interval
        as the target
    allow_target_padding
        flag to allow padding
    pad_value
        value to be used for padding
    use_prediction_features
        flag to indicate if prediction range features should be returned
    prediction_length
        length of the prediction range, must be set if
        use_prediction_features is True
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        instance_length: int,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        allow_target_padding: bool = False,
        pad_value: float = 0.0,
        use_prediction_features: bool = False,
        prediction_length: Optional[int] = None,
    ) -> None:
        self.instance_sampler = instance_sampler
        self.instance_length = instance_length
        self.output_NTC = output_NTC
        self.dynamic_feature_fields = time_series_fields
        self.target_field = target_field
        self.allow_target_padding = allow_target_padding
        self.pad_value = pad_value
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field

        assert (
            not use_prediction_features or prediction_length is not None
        ), "You must specify `prediction_length` if `use_prediction_features`"

        self.use_prediction_features = use_prediction_features
        self.prediction_length = prediction_length

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        ts_fields = self.dynamic_feature_fields + [self.target_field]
        ts_target = data[self.target_field]

        len_target = ts_target.shape[-1]

        sampling_indices = self.instance_sampler(ts_target)

        for i in sampling_indices:
            d = data.copy()

            pad_length = max(self.instance_length - i, 0)

            # update start field
            d[self.start_field] = shift_timestamp(
                data[self.start_field], i - self.instance_length
            )

            # set is_pad field
            is_pad = np.zeros(self.instance_length, dtype=ts_target.dtype)
            if pad_length > 0:
                is_pad[:pad_length] = 1
            d[self.is_pad_field] = is_pad

            # update time series fields
            for ts_field in ts_fields:
                full_ts = data[ts_field]
                if pad_length > 0:
                    pad_pre = self.pad_value * np.ones(
                        shape=full_ts.shape[:-1] + (pad_length,)
                    )
                    past_ts = np.concatenate(
                        [pad_pre, full_ts[..., :i]], axis=-1
                    )
                else:
                    past_ts = full_ts[..., (i - self.instance_length) : i]

                past_ts = past_ts.transpose() if self.output_NTC else past_ts
                d[self._past(ts_field)] = past_ts

                if self.use_prediction_features:
                    if not ts_field == self.target_field:
                        future_ts = full_ts[
                            ..., i : i + self.prediction_length
                        ]
                        future_ts = (
                            future_ts.transpose()
                            if self.output_NTC
                            else future_ts
                        )
                        d[self._future(ts_field)] = future_ts

                del d[ts_field]

            d[self.forecast_start_field] = shift_timestamp(
                d[self.start_field], self.instance_length
            )

            yield d


class ContinuousTimeInstanceSplitter(FlatMapTransformation):
    """
    Selects training instances by slicing "intervals" from a continous-time
    process instantiation. Concretely, the input data is expected to describe an
    instantiation from a point (or jump) process, with the "target"
    identifying inter-arrival times and other features (marks), as described
    in detail below.

    The splitter will then take random points in continuous time from each
    given observation, and return a (variable-length) array of points in
    the past (context) and the future (prediction) intervals.

    The transformation is analogous to its discrete counterpart
    `InstanceSplitter` except that

    - It does not allow "incomplete" records. That is, the past and future
      intervals sampled are always complete
    - Outputs a (T, C) layout.
    - Does not accept `time_series_fields` (i.e., only accepts target fields) as these
      would typically not be available in TPP data.

    The target arrays are expected to have (2, T) layout where the first axis
    corresponds to the (i) interarrival times between consecutive points, in
    order and (ii) integer identifiers of marks (from {0, 1, ..., :code:`num_marks`}).
    The returned arrays will have (T, 2) layout.

    For example, the array below corresponds to a target array where points with timestamps
    0.5, 1.1, and 1.5 were observed belonging to categories (marks) 3, 1 and 0
    respectively: :code:`[[0.5, 0.6, 0.4], [3, 1, 0]]`.

    Parameters
    ----------
    past_interval_length
        length of the interval seen before making prediction
    future_interval_length
        length of the interval that must be predicted
    train_sampler
        instance sampler that provides sampling indices given a time-series
    target_field
        field containing the target
    start_field
        field containing the start date of the of the point process observation
    end_field
        field containing the end date of the point process observation
    forecast_start_field
        output field that will contain the time point where the forecast starts
    """

    def __init__(
        self,
        past_interval_length: float,
        future_interval_length: float,
        instance_sampler: ContinuousTimePointSampler,
        target_field: str = FieldName.TARGET,
        start_field: str = FieldName.START,
        end_field: str = "end",
        forecast_start_field: str = FieldName.FORECAST_START,
    ) -> None:

        assert (
            future_interval_length > 0
        ), "Prediction interval must have length greater than 0."

        self.instance_sampler = instance_sampler
        self.past_interval_length = past_interval_length
        self.future_interval_length = future_interval_length
        self.target_field = target_field
        self.start_field = start_field
        self.end_field = end_field
        self.forecast_start_field = forecast_start_field

    # noinspection PyMethodMayBeStatic
    def _mask_sorted(self, a: np.ndarray, lb: float, ub: float):
        start = np.searchsorted(a, lb)
        end = np.searchsorted(a, ub)
        return np.arange(start, end)

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:

        assert data[self.start_field].freq == data[self.end_field].freq

        total_interval_length = (
            data[self.end_field] - data[self.start_field]
        ) / data[self.start_field].freq.delta

        sampling_times = self.instance_sampler(total_interval_length)

        ia_times = data[self.target_field][0, :]
        marks = data[self.target_field][1:, :]

        ts = np.cumsum(ia_times)
        assert ts[-1] < total_interval_length, (
            "Target interarrival times provided are inconsistent with "
            "start and end timestamps."
        )

        # select field names that will be included in outputs
        keep_cols = {
            k: v
            for k, v in data.items()
            if k not in [self.target_field, self.start_field, self.end_field]
        }

        for future_start in sampling_times:

            r: DataEntry = dict()

            past_start = future_start - self.past_interval_length
            future_end = future_start + self.future_interval_length

            assert past_start >= 0

            past_mask = self._mask_sorted(ts, past_start, future_start)

            past_ia_times = np.diff(np.r_[0, ts[past_mask] - past_start])[
                np.newaxis
            ]

            r[f"past_{self.target_field}"] = np.concatenate(
                [past_ia_times, marks[:, past_mask]], axis=0
            ).transpose()

            r["past_valid_length"] = np.array([len(past_mask)])

            r[self.forecast_start_field] = (
                data[self.start_field]
                + data[self.start_field].freq.delta * future_start
            )

            if is_train:  # include the future only if is_train
                assert future_end <= total_interval_length

                future_mask = self._mask_sorted(ts, future_start, future_end)

                future_ia_times = np.diff(
                    np.r_[0, ts[future_mask] - future_start]
                )[np.newaxis]

                r[f"future_{self.target_field}"] = np.concatenate(
                    [future_ia_times, marks[:, future_mask]], axis=0
                ).transpose()

                r["future_valid_length"] = np.array([len(future_mask)])

            # include other fields
            r.update(keep_cols.copy())

            yield r
