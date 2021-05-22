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

from collections import Counter
from typing import Iterator, List, Optional

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    InstanceSplitter,
    MapTransformation,
    shift_timestamp,
    target_transformation_length,
)


class BroadcastTo(MapTransformation):
    @validated()
    def __init__(
        self,
        field: str,
        ext_length: int = 0,
        target_field: str = FieldName.TARGET,
    ) -> None:
        self.field = field
        self.ext_length = ext_length
        self.target_field = target_field

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.ext_length, is_train
        )
        data[self.field] = np.broadcast_to(
            data[self.field],
            (data[self.field].shape[:-1] + (length,)),
        )
        return data


class TFTInstanceSplitter(InstanceSplitter):
    @validated()
    def __init__(
        self,
        instance_sampler,
        past_length: int,
        future_length: int,
        target_field: str = FieldName.TARGET,
        is_pad_field: str = FieldName.IS_PAD,
        start_field: str = FieldName.START,
        forecast_start_field: str = FieldName.FORECAST_START,
        observed_value_field: str = FieldName.OBSERVED_VALUES,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: Optional[List[str]] = None,
        past_time_series_fields: Optional[List[str]] = None,
        dummy_value: float = 0.0,
    ) -> None:

        assert past_length > 0, "The value of `past_length` should be > 0"
        assert future_length > 0, "The value of `future_length` should be > 0"

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.dummy_value = dummy_value

        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.observed_value_field = observed_value_field

        self.ts_fields = time_series_fields or []
        self.past_ts_fields = past_time_series_fields or []

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pl = self.future_length
        lt = self.lead_time
        target = data[self.target_field]

        sampled_indices = self.instance_sampler(target)

        slice_cols = (
            self.ts_fields
            + self.past_ts_fields
            + [self.target_field, self.observed_value_field]
        )
        for i in sampled_indices:
            pad_length = max(self.past_length - i, 0)
            d = data.copy()

            for field in slice_cols:
                if i >= self.past_length:
                    past_piece = d[field][..., i - self.past_length : i]
                else:
                    pad_block = np.full(
                        shape=d[field].shape[:-1] + (pad_length,),
                        fill_value=self.dummy_value,
                        dtype=d[field].dtype,
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[field][..., :i]], axis=-1
                    )
                future_piece = d[field][..., (i + lt) : (i + lt + pl)]
                if field in self.ts_fields:
                    piece = np.concatenate([past_piece, future_piece], axis=-1)
                    if self.output_NTC:
                        piece = piece.transpose()
                    d[field] = piece
                else:
                    if self.output_NTC:
                        past_piece = past_piece.transpose()
                        future_piece = future_piece.transpose()
                    if field not in self.past_ts_fields:
                        d[self._past(field)] = past_piece
                        d[self._future(field)] = future_piece
                        del d[field]
                    else:
                        d[field] = past_piece
            pad_indicator = np.zeros(self.past_length)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1
            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = shift_timestamp(
                d[self.start_field], i + lt
            )
            yield d
