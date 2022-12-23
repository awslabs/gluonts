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
from typing import Optional

import pandas as pd

from .common import DataEntry
from .field_names import FieldName


def forecast_start(entry, time_axis: int = -1):
    return entry[FieldName.START] + entry[FieldName.TARGET].shape[time_axis]


def period_index(entry: DataEntry, freq=None) -> pd.PeriodIndex:
    if freq is None:
        freq = entry[FieldName.START].freq

    return pd.period_range(
        start=entry[FieldName.START],
        periods=entry[FieldName.TARGET].shape[-1],
        freq=freq,
    )


def to_pandas(entry: DataEntry, freq: Optional[str] = None) -> pd.Series:
    """
    Transform a dictionary into a pandas.Series object, using its "start" and
    "target" fields.

    Parameters
    ----------
    entry
        Dictionary containing the time series data.
    freq
        Frequency to use in the pandas.Series index.

    Returns
    -------
    pandas.Series
        Pandas time series object.
    """
    return pd.Series(
        entry[FieldName.TARGET],
        index=period_index(entry, freq=freq),
    )
