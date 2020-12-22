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

from typing import Iterator, List, Optional

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.transform import (
    FlatMapTransformation,
    InstanceSampler,
    MapTransformation,
    shift_timestamp,
)


class LeadtimeShifter(MapTransformation):
    @validated()
    def __init__(
        self,
        lead_time: int,
        target_field: str = "target",
        time_series_fields: Optional[List[str]] = None,
    ) -> None:
        assert lead_time >= 0

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
        d[self._input_target] = d[self.target_field][
            ..., : -(self.lead_time + 1)
        ]
        d[self._label_target] = d[self.target_field][
            ..., (self.lead_time + 1) :
        ]
        del d[self.target_field]
        for ts_field in self.time_series_fields:
            d[ts_field] = d[ts_field][..., (self.lead_time + 1) :]
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


class AnomalyScoringSplitter(FlatMapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        train_sampler: InstanceSampler,
        train_window_length: int,
        output_NTC: bool = True,
        time_series_fields: Optional[List[str]] = None,
        pick_incomplete: bool = True,
    ) -> None:
        assert train_window_length > 0
        self.train_window_length = train_window_length
        self.train_sampler = train_sampler
        self.output_NTC = output_NTC
        self.ts_fields = (
            time_series_fields if time_series_fields is not None else []
        )
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.pick_incomplete = pick_incomplete

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        slice_cols = self.ts_fields + [self.target_field]
        target = data[self.target_field]

        len_target = target.shape[-1]
        if is_train and len_target == 0:
            return

        if not is_train:
            assert len_target > 0

        d = data.copy()
        if not is_train:
            for ts_field in slice_cols:
                assert d[ts_field].shape[-1] >= len_target
                d[ts_field] = d[ts_field][..., :len_target]
                if self.output_NTC:
                    d[ts_field] = d[ts_field].transpose()
            d[self.forecast_start_field] = d[self.start_field]
            yield d
            return

        if self.pick_incomplete:
            sampling_indices = self.train_sampler(target, 0, len_target)
        elif len_target < self.train_window_length:
            return
        else:
            sampling_indices = self.train_sampler(
                target, self.train_window_length, len_target
            )

        for i in sampling_indices:
            pad_length = max(self.train_window_length - i, 0)
            if not self.pick_incomplete:
                assert pad_length == 0
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
