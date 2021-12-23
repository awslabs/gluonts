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
from typing import Callable, List, cast

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Tick

from .common import DataEntry


class ProcessStartField:
    def __init__(self, name: str, freq: str) -> None:
        self.name = name
        self.freq = freq

    def __call__(self, data: DataEntry) -> DataEntry:
        try:
            value = ProcessStartField.process(data[self.name], self.freq)
        except (TypeError, ValueError) as e:
            raise Exception(f'Error "{e}" occurred when reading field "{self.name}"')

        data[self.name] = value

        return data

    @staticmethod
    @lru_cache(maxsize=10000)
    def process(string: str, freq: str) -> pd.Timestamp:
        timestamp = pd.Timestamp(string, freq=freq)

        # operate on time information (days, hours, minute, second)
        if isinstance(timestamp.freq, Tick):
            return pd.Timestamp(timestamp.floor(timestamp.freq), timestamp.freq)

        # since we are only interested in the data piece, we normalize the
        # time information
        timestamp = timestamp.replace(
            hour=0, minute=0, second=0, microsecond=0, nanosecond=0
        )

        return timestamp.freq.rollforward(timestamp)


class ProcessTimeSeriesField:
    def __init__(self, name, is_required: bool, is_static: bool, is_cat: bool) -> None:
        self.name = name
        self.is_required = is_required
        self.req_ndim = 1 if is_static else 2
        self.dtype = np.int64 if is_cat else np.float32

    def __call__(self, data: DataEntry) -> DataEntry:
        value = data.get(self.name, None)

        if value is not None:
            value = np.asarray(value, dtype=self.dtype)
            dim_diff = self.req_ndim - value.ndim
            if dim_diff == 1:
                value = np.expand_dims(a=value, axis=0)
            elif dim_diff != 0:
                raise Exception(
                    f"JSON array has bad shape - expected {self.req_ndim} dimensions got {dim_diff}"
                )

            data[self.name] = value
            return data
        elif not self.is_required:
            return data
        else:
            raise Exception(f"JSON object is missing a required field `{self.name}`")


class ProcessDataEntry:
    def __init__(self, freq: str, one_dim_target: bool = True) -> None:
        self.trans = cast(
            List[Callable[[DataEntry], DataEntry]],
            [
                ProcessStartField("start", freq=freq),
                ProcessTimeSeriesField(
                    "target", is_required=True, is_cat=False, is_static=one_dim_target
                ),
                ProcessTimeSeriesField(
                    "feat_dynamic_cat", is_required=False, is_cat=True, is_static=False
                ),
                ProcessTimeSeriesField(
                    "feat_dynamic_real",
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    "feat_static_cat", is_required=False, is_cat=True, is_static=True
                ),
                ProcessTimeSeriesField(
                    "feat_static_real", is_required=False, is_cat=False, is_static=True
                ),
            ],
        )

    def __call__(self, data: DataEntry) -> DataEntry:
        for t in self.trans:
            data = t(data)
        return data
