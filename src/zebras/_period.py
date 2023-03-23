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

"""
``zebras.Period`` and ``zebras.Periods`` are classes to track points in time
with periodicity. They provide similar functionality to ``pandas.Period`` and
``pandas.PeriodIndex`` but offer some advantages:

* more consistent API
* improved ergonomics
* better performance

Both classes are just thin layers around ``numpy.datetime64`` objects, but
provide a more usable interface.

Both classes have easier to use factory-functions in ``zebras.period`` and
``zebras.periods``, akin to how ``numpy.ndarray`` objects are usually
constructed using ``numpy.array``.

While ``zebras.Period`` represents a single time-stamp, ``zebras.Periods`` are
a set of equidistant timestamps where the gap between consecutive timestamps
is the period.


``py
first = zb.period("2020-01", "3M")
index = zb.periods("2020-01", "3M", 3)

first.periods(3) == index
first == index[0]
```
"""

from __future__ import annotations

import datetime
import functools
import re
from dataclasses import dataclass
from typing import Any, Union, cast, overload

import numpy as np
from dateutil.parser import parse as du_parse

from gluonts.core import serde

from ._freq import Freq


def _is_number(value):
    return isinstance(value, (int, np.integer))


class _BasePeriod:
    data: Any
    freq: Freq

    @property
    def year(self) -> np.ndarray:
        return self.data.astype("M8[Y]").astype(int) + 1970

    @property
    def month(self) -> np.ndarray:
        return self.data.astype("M8[M]").astype(int) % 12 + 1

    @property
    def day(self) -> np.ndarray:
        return (self.data.astype("M8[D]") - self.data.astype("M8[M]")).astype(
            int
        ) + 1

    @property
    def hour(self) -> np.ndarray:
        return (self.data.astype("M8[h]") - self.data.astype("M8[D]")).astype(
            int
        )

    @property
    def minute(self) -> np.ndarray:
        return (self.data.astype("M8[m]") - self.data.astype("M8[h]")).astype(
            int
        )

    @property
    def second(self) -> np.ndarray:
        return (self.data.astype("M8[s]") - self.data.astype("M8[m]")).astype(
            int
        )

    @property
    def dayofweek(self) -> np.ndarray:
        return (self.data.astype("M8[D]").astype(int) - 4) % 7

    @property
    def dayofyear(self) -> np.ndarray:
        return (self.data.astype("M8[D]") - self.data.astype("M8[Y]")).astype(
            int
        ) + 1

    @property
    def week(self) -> np.ndarray:
        # Note: In Python 3.9 `isocalendar()` returns a named tuple, but we
        # need to support 3.7 and 3.8, so we use index one for the week.
        return np.array(
            [
                cal.isocalendar()[1]
                for cal in self.data.astype(datetime.datetime)
            ]
        )

    def __add__(self, other):
        if _is_number(other):
            return self.__class__(
                self.freq.shift(self.data, other),
                self.freq,
            )

    def __sub__(self, other):
        if _is_number(other):
            return self.__class__(
                self.freq.shift(self.data, -other),
                self.freq,
            )

        else:
            return self.data - other.data

        raise ValueError(other)


@functools.total_ordering
@dataclass
class Period(_BasePeriod):
    data: np.datetime64
    freq: Freq

    @property
    def __init_passed_kwargs__(self) -> dict:
        return {"data": self.data, "freq": self.freq}

    def periods(self, count: int):
        return Periods(
            self.freq.range(self.data, count),
            self.freq,
        )

    def to_pandas(self):
        import pandas as pd

        return pd.Period(self.data.astype(object), self.freq.to_pandas())

    def to_timestamp(self):
        return self.data.astype(object)

    def __repr__(self) -> str:
        return f"Period<{self.data}, {self.freq}>"

    def __lt__(self, other: Period) -> bool:
        # convert numpy.bool_ into bool
        return cast(bool, self.data < other.data)

    def to_numpy(self) -> np.datetime64:
        return self.data

    def __array__(self) -> np.datetime64:
        return self.data


@dataclass
class Periods(_BasePeriod):
    data: np.ndarray
    freq: Freq

    @property
    def start(self) -> Period:
        return self[0]

    @property
    def end(self) -> Period:
        """
        Last timestamp.

        >>> p = periods("2021", "D", 365)
        >>> assert p.end == period("2021-12-31", "D")

        """

        return self[-1]

    def head(self, count: int) -> Periods:
        """
        First ``count`` timestamps.

        >>> p = periods("2021", "D", 365)
        >>> assert p.head(5) == periods("2021-01-01", "D", 5)

        """

        return self[:count]

    def tail(self, count: int) -> Periods:
        """
        Last ``count`` timestamps.

        >>> p = periods("2021", "D", 365)
        >>> assert p.tail(5) == periods("2021-12-27", "D", 5)

        """

        return self[-count:]

    def future(self, count: int) -> Periods:
        """
        Next ``count`` timestamps.

        >>> p = periods("2021", "D", 365)
        >>> assert p.future(5) == periods("2022-01-01", "D", 5)

        """
        return (self.end + 1).periods(count)

    def past(self, count: int) -> Periods:
        """
        Previous ``count`` timestamps.

        >>> p = periods("2021", "D", 365)
        >>> assert p.past(5) == periods("2020-12-27", "D", 5)

        """

        return (self.start - count).periods(count)

    def prepend(self, count: int) -> Periods:
        """
        Copy which contains ``count`` past timestamps.

        >>> p = periods("2021", "D", 365)
        >>> assert p.prepend(5) == periods("2020-12-27", "D", 370)

        """
        return Periods(
            np.concatenate([self.past(count).data, self.data]),
            self.freq,
        )

    def extend(self, count: int) -> Periods:
        """
        Copy which contains ``count`` future timestamps.

        >>> p = periods("2021", "D", 365)
        >>> assert p.extend(5) == periods("2021", "D", 370)

        """
        return Periods(
            np.concatenate([self.data, self.future(count).data]),
            self.freq,
        )

    def to_pandas(self):
        import pandas as pd

        # older versions of pandas expect ns-datetime64
        return pd.PeriodIndex(
            self.data.astype("M8[ns]"), freq=self.freq.to_pandas()
        )

    def intersection(self, other):
        # TODO: Is this needed?
        return self.data[np.in1d(self, other)]

    def index_of(self, period: Union[str, Period]):
        """
        Return the index of ``period``

        >>> p = periods("2021", "D", 365)
        >>> assert p.index_of(period("2021-02-01", "D")) == 31

        """

        if isinstance(period, str):
            period = Period(
                np.datetime64(du_parse(period), self.freq.np_freq), self.freq
            )

        idx = (period - self.start).astype(int) // self.freq.step
        assert 0 <= idx < len(self), idx

        return idx

    def __len__(self):
        return len(self.data)

    @overload
    def __getitem__(self, idx: int) -> Period:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Periods:
        ...

    def __getitem__(self, idx):
        if _is_number(idx):
            return Period(self.data[idx], self.freq)

        return Periods(self.data[idx], self.freq)

    def __eq__(self, other):
        if not isinstance(other, Periods):
            return False

        return self.freq.n == other.freq.n and np.array_equal(
            self.data, other.data
        )

    def to_numpy(self) -> np.ndarray:
        return self.data

    def __array__(self) -> np.ndarray:
        return self.data


@serde.encode.register
def _encode_zebras_periods(v: Periods):
    return {
        "__kind__": "instance",
        "class": "zebras.periods",
        "kwargs": serde.encode(
            {"start": v.start, "freq": str(v.freq), "count": len(v)}
        ),
    }


def period(data, freq=None) -> Period:
    if hasattr(data, "freqstr") and freq is None:
        freqstr = data.freqstr
        freq = Freq.from_pandas(data.freqstr)
    else:
        freqstr = freq
        freq = Freq.from_pandas(freq)

    if isinstance(data, Period):
        data = data.data

    if isinstance(data, str):
        data = du_parse(
            data,
            default=datetime.datetime(1970, 1, 1),
            ignoretz=True,
        )

    if freq.name == "W":
        period = Period(np.datetime64(data, freq.np_freq), freq)
        if re.match(r"W-\w{3}", freqstr):
            weekday_offset = get_weekday_offset(freqstr)
        else:
            weekday_offset = 0
        period.data -= (cast(int, period.dayofweek) - weekday_offset) % 7
        return period

    return Period(np.datetime64(data, freq.np_freq), freq)


def get_weekday_offset(freqstr):
    day_offsets = {
        "MON": 0,
        "TUE": 1,
        "WED": 2,
        "THU": 3,
        "FRI": 4,
        "SAT": 5,
        "SUN": 6,
    }

    day = re.findall(r"W-\w{3}", freqstr)[0].split("-")[1]
    return day_offsets[day]


def periods(start, freq, count: int) -> Period:
    return period(start, freq).periods(count)
