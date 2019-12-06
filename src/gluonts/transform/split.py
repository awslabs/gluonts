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

from ._base import FlatMapTransformation
from .sampler import InstanceSampler


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
    train_sampler
        instance sampler that provides sampling indices given a time-series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout
    time_series_fields
        fields that contains time-series, they are split in the same interval
        as the target
    pick_incomplete
        whether training examples can be sampled with only a part of
        past_length time-units
        present for the time series. This is useful to train models for
        cold-start. In such case, is_pad_out contains an indicator whether
        data is padded or not.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        train_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        output_NTC: bool = True,
        time_series_fields: Optional[List[str]] = None,
        pick_incomplete: bool = True,
    ) -> None:

        assert future_length > 0

        self.train_sampler = train_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.output_NTC = output_NTC
        self.ts_fields = (
            time_series_fields if time_series_fields is not None else []
        )
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.pick_incomplete = pick_incomplete

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pl = self.future_length
        slice_cols = self.ts_fields + [self.target_field]
        target = data[self.target_field]

        len_target = target.shape[-1]

        if is_train:
            if len_target < self.future_length:
                # We currently cannot handle time series that are shorter than
                # the prediction length during training, so we just skip these.
                # If we want to include them we would need to pad and to mask
                # the loss.
                sampling_indices: List[int] = []
            else:
                if self.pick_incomplete:
                    sampling_indices = self.train_sampler(
                        target, 0, len_target - self.future_length
                    )
                else:
                    sampling_indices = self.train_sampler(
                        target,
                        self.past_length,
                        len_target - self.future_length,
                    )
        else:
            sampling_indices = [len_target]
        for i in sampling_indices:
            pad_length = max(self.past_length - i, 0)
            if not self.pick_incomplete:
                assert pad_length == 0
            d = data.copy()
            for ts_field in slice_cols:
                if i > self.past_length:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.past_length : i]
                elif i < self.past_length:
                    pad_block = np.zeros(
                        d[ts_field].shape[:-1] + (pad_length,),
                        dtype=d[ts_field].dtype,
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                else:
                    past_piece = d[ts_field][..., :i]
                d[self._past(ts_field)] = past_piece
                d[self._future(ts_field)] = d[ts_field][..., i : i + pl]
                del d[ts_field]
            pad_indicator = np.zeros(self.past_length)
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
                d[self.start_field], i
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

        if is_train:
            if len_target < self.instance_length:
                sampling_indices = (
                    # Returning [] for all time series will cause this to be in loop forever!
                    [len_target]
                    if self.allow_target_padding
                    else []
                )
            else:
                sampling_indices = self.instance_sampler(
                    ts_target, self.instance_length, len_target
                )
        else:
            sampling_indices = [len_target]

        for i in sampling_indices:
            d = data.copy()

            pad_length = max(self.instance_length - i, 0)

            # update start field
            d[self.start_field] = shift_timestamp(
                data[self.start_field], i - self.instance_length
            )

            # set is_pad field
            is_pad = np.zeros(self.instance_length)
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

                if self.use_prediction_features and not is_train:
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
