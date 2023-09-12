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

from gluonts.meta.export import re_export


__all__ = [
    "batch",
    "from_pandas",
    *re_export(
        __name__,
        "schema",
        _freq=["Freq", "freq"],
        _period=["Period", "Periods", "period", "periods"],
        _time_frame=["BatchTimeFrame", "TimeFrame", "time_frame"],
        _split_frame=["BatchSplitFrame", "SplitFrame", "split_frame"],
        _time_series=["time_series", "BatchTimeSeries", "TimeSeries"],
    ),
]

from typing import TypeVar

from . import TimeSeries, TimeFrame, SplitFrame, Periods, Freq  # type: ignore

Batchable = TypeVar("Batchable", TimeSeries, TimeFrame, SplitFrame)


def batch(xs: list):
    assert xs, "Passed data cannot be empty."
    types = set(map(type, xs))
    assert (
        len(types) == 1
    ), "All values need to be of same type, got: " + ", ".join(
        f"'{ty.__name__}'" for ty in types
    )
    ty = types.pop()
    assert ty in set(
        Batchable.__constraints__  # type: ignore
    ), f"Unsupported type: '{ty.__name__}'"

    return ty._batch(xs)  # type: ignore


def from_pandas(obj):
    """Convert pandas offsets, date indices and data frames into ``zebras``
    equivalents."""
    import pandas as pd
    from pandas.core.base import IndexOpsMixin
    from pandas.tseries.offsets import BaseOffset

    if isinstance(obj, pd.DataFrame):
        return TimeFrame.from_pandas(obj)

    if isinstance(obj, IndexOpsMixin):
        return Periods.from_pandas(obj)

    if isinstance(obj, BaseOffset):
        return Freq.from_pandas(obj)

    raise ValueError(f"Can not convert value of type {type(obj).__name__}")
