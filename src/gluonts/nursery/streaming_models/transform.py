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

import logging
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from gluonts.core.component import DType, validated
from gluonts.dataset.common import DataEntry
from gluonts.transform import (
    FlatMapTransformation,
    InstanceSampler,
    MapTransformation,
    SimpleTransformation,
    shift_timestamp,
)

from .native import ema

logger = logging.getLogger(__name__)


class AddStreamScale(SimpleTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        scale_field: str,
        state_field: str,
        minimum_value: float,
        initial_scale: float,
        alpha: float,
        mean_field: str,
        normalization: str,
    ) -> None:
        self.target_field = target_field
        self.scale_field = scale_field
        self.state_field = state_field
        self.minimum_value = minimum_value
        self.initial_scale = initial_scale
        self.alpha = alpha
        self.mean_field = mean_field
        self.normalization = normalization

    def transform(self, data: DataEntry) -> DataEntry:
        target = data[self.target_field]
        state = data.get(self.state_field)

        scale, mean, var, new_state = ema(
            target,
            alpha=self.alpha,
            minimum_value=self.minimum_value,
            initial_scale=self.initial_scale,
            state=state,
        )

        if self.normalization == "centeredscale":
            data[self.scale_field] = scale
        elif self.normalization == "standardization":
            data[self.scale_field] = var
        data[self.mean_field] = mean
        data[self.state_field] = np.array(new_state)
        return data


class AddLags(SimpleTransformation):
    @validated()
    def __init__(
        self,
        lag_seq: List[int],
        lag_field: str,
        target_field: str,
        lag_state_field: str,
    ) -> None:
        self.lag_seq = sorted(lag_seq)
        self.lag_field = lag_field
        self.target_field = target_field
        self.lag_state_field = lag_state_field
        self.max_lag = self.lag_seq[-1]

    def transform(self, data: DataEntry) -> DataEntry:
        target = data[self.target_field]
        buffer = data.get(self.lag_state_field)
        if buffer is None:
            t = np.concatenate([np.zeros(self.max_lag), target])
        else:
            t = np.concatenate([buffer, target])
        lags = np.vstack(
            [t[self.max_lag - l : len(t) - l] for l in self.lag_seq]
        )
        data[self.lag_field] = np.nan_to_num(lags)
        data[self.lag_state_field] = t[-self.max_lag :]
        return data


class AddStreamAggregateLags(SimpleTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        lag_state_field: str,
        lead_time: int,
        base_freq: str,
        agg_freq: str,
        agg_lags: List[int],
        agg_fun: str = "mean",
        dtype: DType = np.float32,
    ) -> None:

        self.target_field = target_field
        self.feature_name = output_field
        self.lead_time = lead_time
        self.base_freq = base_freq
        self.agg_freq = agg_freq
        self.agg_lags = agg_lags
        self.agg_fun = agg_fun
        self.lag_state_field = lag_state_field
        self.dtype = dtype

        self.ratio = pd.Timedelta(self.agg_freq) / pd.Timedelta(self.base_freq)
        assert (
            self.ratio.is_integer() and self.ratio >= 1
        ), "The aggregate frequency should be a multiple of the base frequency."
        self.ratio = int(self.ratio)

        # convert lags to original freq and adjust based on lead time
        adj_lags = [
            lag * self.ratio - (self.lead_time - 1) for lag in self.agg_lags
        ]

        self.half_window = (self.ratio - 1) // 2
        valid_adj_lags = [x for x in adj_lags if x - self.half_window > 0]
        self.valid_lags = [
            int(np.ceil(x / self.ratio)) for x in valid_adj_lags
        ]
        self.offset = (self.lead_time - 1) % self.ratio

        assert len(self.valid_lags) > 0

        if len(self.agg_lags) - len(self.valid_lags) > 0:
            logger.info(
                f"The aggregate lags {set(self.agg_lags[:- len(self.valid_lags)])} "
                f"of frequency {self.agg_freq} are ignored."
            )

        self.max_state_lag = max(self.valid_lags)

    def transform(self, data: DataEntry) -> DataEntry:
        assert self.base_freq == data["start"].freq

        buffer = data.get(self.lag_state_field)
        if buffer is None:
            t = data[self.target_field]
            t_agg = (pd.Series(t).rolling(self.ratio).agg(self.agg_fun))[
                self.ratio - 1 :
            ]
        else:
            t = np.concatenate(
                [buffer["base_target"], data[self.target_field]]
            )
            new_agg_lags = (
                pd.Series(t).rolling(self.ratio).agg(self.agg_fun)
            )[self.ratio - 1 :]
            t_agg = pd.Series(
                np.concatenate([buffer["agg_lags"], new_agg_lags.values])
            )

        # compute the aggregate lags for each time point of the time series
        agg_vals = np.concatenate(
            [
                np.zeros(
                    (max(self.valid_lags) * self.ratio + self.half_window,)
                ),
                t_agg.values,
            ],
            axis=0,
        )
        lags = np.vstack(
            [
                agg_vals[
                    -(
                        l * self.ratio
                        - self.offset
                        - self.half_window
                        + len(data[self.target_field])
                        - 1
                    ) : -(l * self.ratio - self.offset - self.half_window - 1)
                    if -(l * self.ratio - self.offset - self.half_window - 1)
                    is not 0
                    else None
                ]
                for l in self.valid_lags
            ]
        )

        # update the data entry
        data[self.feature_name] = np.nan_to_num(lags).astype(self.dtype)
        data[self.lag_state_field] = {
            "agg_lags": t_agg.values[-self.max_state_lag * self.ratio + 1 :],
            "base_target": t[-self.ratio + 1 :] if self.ratio > 1 else [],
        }

        assert data[self.feature_name].shape == (
            len(self.valid_lags),
            len(data[self.target_field]),
        )

        return data


class CopyField(SimpleTransformation):
    """
    Copies the value of input_field into output_field and does nothing
    if input_field is not present or None.
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        input_field: str,
    ) -> None:
        self.output_field = output_field
        self.input_field = input_field

    def transform(self, data: DataEntry) -> DataEntry:

        field = data.get(self.input_field)
        if field is not None:
            data[self.output_field] = data[self.input_field].copy()

        return data


class LeadtimeShifter(MapTransformation):
    @validated()
    def __init__(
        self,
        lead_time: int,
        target_field: str = "target",
        time_series_fields: Optional[List[str]] = None,
    ) -> None:
        self.lead_time = lead_time
        self.target_field = target_field
        self.time_series_fields = (
            time_series_fields if time_series_fields is not None else []
        )

        self._input_target = f"input_{self.target_field}"
        self._label_target = f"label_{self.target_field}"

    def map_transform(self, d: DataEntry, is_train: bool) -> DataEntry:
        if not is_train:
            return d
        d[self._input_target] = d[self.target_field][..., : -self.lead_time]
        d[self._label_target] = d[self.target_field][..., self.lead_time :]
        del d[self.target_field]
        for ts_field in self.time_series_fields:
            d[ts_field] = d[ts_field][..., self.lead_time :]
        return d


class AddShiftedTimestamp(MapTransformation):
    """
    A transformation that adds a timestamp field relative to an existing one.

    Parameters
    ----------
    input_field
        The field where the start timestamp of the data entry can be found.
    output_field
        The field name where to store the forecast start timestamp.
    shift
        How much to shift the start timestamp to get the forecast start.
    """

    @validated()
    def __init__(
        self, input_field: str, output_field: str, shift: int
    ) -> None:
        assert shift >= 1

        self.shift = shift
        self.input_field = input_field
        self.output_field = output_field

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data[self.output_field] = shift_timestamp(
            data[self.input_field], self.shift
        )
        return data


class StreamingInstanceSplitter(FlatMapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        train_window_length: int,
        output_NTC: bool = True,
        time_series_fields: Optional[List[str]] = None,
    ) -> None:
        assert train_window_length > 0
        self.train_window_length = train_window_length
        self.instance_sampler = instance_sampler
        self.output_NTC = output_NTC
        self.ts_fields = (
            time_series_fields if time_series_fields is not None else []
        )
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        slice_cols = self.ts_fields + [self.target_field]
        target = data[self.target_field]

        sampling_indices = self.instance_sampler(target)

        for i in sampling_indices:
            pad_length = max(self.train_window_length - i, 0)
            d = data.copy()
            for ts_field in slice_cols:
                if i > self.train_window_length:
                    # truncate to past_length
                    piece = d[ts_field][..., i - self.train_window_length : i]
                elif i < self.train_window_length:
                    pad_block = np.zeros(
                        d[ts_field].shape[:-1] + (pad_length,),
                        dtype=d[ts_field].dtype,
                    )
                    piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                else:
                    piece = d[ts_field][..., :i]
                d[ts_field] = piece

            pad_indicator = np.zeros(self.train_window_length)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            if self.output_NTC:
                for ts_field in slice_cols:
                    d[ts_field] = d[ts_field].transpose()

            d[self.is_pad_field] = pad_indicator
            d[self.forecast_start_field] = shift_timestamp(
                d[self.start_field], i
            )
            yield d
