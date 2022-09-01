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

from .field_names import FieldName


def forecast_start(entry, time_axis: int = -1):
    return entry[FieldName.START] + entry[FieldName.TARGET].shape[time_axis]


def to_pandas(instance: dict, freq: Optional[str] = None) -> pd.Series:
    """
    Transform a dictionary into a pandas.Series object, using its "start" and
    "target" fields.

    Parameters
    ----------
    instance
        Dictionary containing the time series data.
    freq
        Frequency to use in the pandas.Series index.

    Returns
    -------
    pandas.Series
        Pandas time series object.
    """
    target = instance[FieldName.TARGET]
    start = instance[FieldName.START]
    if not freq:
        freq = start.freqstr

    return pd.Series(
        target,
        index=pd.period_range(start=start, periods=len(target), freq=freq),
    )
