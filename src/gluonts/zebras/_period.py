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
from dataclasses import dataclass, asdict

import numpy as np
from dateutil.parser import parse as du_parse

from ._freq import Freq


def _is_number(value):
    return isinstance(value, (int, np.integer))


@dataclass
class _BasePeriod:
    data: np.datetime64
    freq: Freq

    @property
    def year(self):
        return self.data.astype("M8[Y]").astype(int) + 1970

    @property
    def month(self):
        return self.data.astype("M8[M]").astype(int) % 12 + 1

    @property
    def day(self):
        return (self.data - self.data.astype("M8[M]")).astype(int) + 1

    @property
    def hour(self):
        return (self.data - self.data.astype("M8[D]")).astype(int)

    @property
    def minute(self):
        return (self.data - self.data.astype("M8[h]")).astype(int)

    @property
    def second(self):
        return (self.data - self.data.astype("M8[m]")).astype(int)

    @property
    def dayofweek(self):
        return (self.data.astype("M8[D]").astype(int) - 4) % 7

    @property
    def dayofyear(self):
        return (self.data.astype("M8[D]") - self.data.astype("M8[Y]")).astype(
            int
        ) + 1

    @property
    def week(self):
        # Note: In Python 3.9 `isocalendar()` returns a named tuple, but we
        # need to support 3.7 and 3.8, so we use index one for the week.
        return np.array(
            [
                cal.isocalendar()[1]
                for cal in self.data.astype(datetime.datetime)
            ]
        )

    @property
    def __init_passed_kwargs__(self):
        return asdict(self)

    def to_numpy(self):
        return self.data

    def __array__(self):
        return self.data

    def __add__(self, other):
        if _is_number(other):
            return self.__class__(
                self.data + self.freq.multiple * other,
                self.freq,
            )

    def __sub__(self, other):
        if _is_number(other):
            return self.__class__(
                self.data - self.freq.multiple * other,
                self.freq,
            )
        else:
            return self.data - other.data

        raise ValueError(other)


@functools.total_ordering
class Period(_BasePeriod):
    def periods(self, count: int):
        return Periods(
            np.arange(
                self.data, count * self.freq.multiple, self.freq.multiple
            ),
            self.freq,
        )

    def to_pandas(self):
        import pandas as pd

        return pd.Period(self.data, self.freq.to_pandas())

    def to_timestamp(self):
        return self.data.astype(object)

    def __repr__(self):
        return f"Period<{self.data}, {self.freq}>"

    def __lt__(self, other: Period):
        return self.data < other.data


class BusinessDay(Period):
    def __add__(self, other):
        if _is_number(other):
            return BusinessDay(
                np.busday_offset(self.data, self.freq.multiple * other),
                self.freq,
            )

    def periods(self, count: int):
        # We first collect all days, even non business days to then filter for
        # business days, of which we then take, each freq.multiple one.
        periods = np.arange(
            self.data, np.busday_offset(self.data, count * self.freq.multiple)
        )
        periods = periods[np.is_busday(periods)]
        periods = periods[:: self.freq.multiple]

        return Periods(periods, self.freq)

    def __repr__(self):
        return f"BusinessDay<{self.data}, {self.freq}>"


class Periods(_BasePeriod):
    @property
    def start(self) -> Period:
        return self[0]

    @property
    def end(self) -> Period:
        return self[-1]

    def head(self, count: int) -> Periods:
        return self[:count]

    def tail(self, count: int) -> Periods:
        return self[-count:]

    def future(self, count: int) -> Periods:
        return (self.end + 1).periods(count)

    def past(self, count: int) -> Periods:
        return (self.start - count).periods(count)

    def prepend(self, count: int) -> Periods:
        return Periods(
            np.concatenate([self.past(count).data, self.data]),
            self.freq,
        )

    def extend(self, count: int) -> Periods:
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
        return self.data[np.in1d(self, other)]

    def index_of(self, period: Period):
        idx = (period - self.start).astype(int)
        assert 0 <= idx < len(self)

        return idx

    def cat(self, other):
        return Periods(np.concatenate([self.data, other.data]), self.freq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Periods:
        if _is_number(idx):
            return Period(self.data[idx], self.freq)

        return Periods(self.data[idx], self.freq)

    def __eq__(self, other):
        if not isinstance(other, Periods):
            return False

        return self.freq.multiple == other.multiple and np.array_equal(
            self.data, other.data
        )


def period(data, freq=None) -> Period:
    if hasattr(data, "freqstr") and freq is None:
        freq = Freq.from_pandas(data.freqstr)
    else:
        freq = Freq.from_pandas(freq)

    if isinstance(data, Period):
        data = data.data

    if isinstance(data, str):
        data = du_parse(
            data,
            default=datetime.datetime(1970, 1, 1),
            ignoretz=True,
        )

    if freq.pd_freq == "B":
        return BusinessDay(np.datetime64(data, freq.np_freq), freq)
    elif freq.pd_freq == "W":
        period = Period(np.datetime64(data, freq.np_freq), freq)
        period.data -= period.dayofweek
        return period

    return Period(np.datetime64(data, freq.np_freq), freq)


def periods(start, freq, count: int) -> Period:
    return period(start, freq).periods(count)
