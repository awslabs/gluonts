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
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from dateutil.parser import parse as du_parse

from gluonts import maybe

from gluonts.itertools import inverse

NpFreq = Tuple[str, int]


def _is_number(value):
    return isinstance(value, (int, np.integer))


def _canonical_freqstr(n: int, freq: str):
    if n == 1:
        return freq

    return f"{n}{freq}"


@dataclass
class Freq:
    np_freq: NpFreq
    multiple: int

    _freq_numpy_to_pandas = {
        "Y": "Y",
        "D": "D",
        "W": "W",
        "M": "M",
        "h": "H",
        "m": "T",
        "s": "S",
    }

    _freq_pandas_to_numpy = dict(
        inverse(_freq_numpy_to_pandas),
        **{
            "A": ("Y", 1),
            "AS": ("Y", 1),
            "YS": ("Y", 1),
            "MS": ("M", 1),
            "MIN": ("m", 1),
            "Q": ("M", 3),
        },
    )

    @property
    def __init_passed_kwargs__(self):
        return asdict(self)

    @classmethod
    def from_pandas(cls, freq):
        if not isinstance(freq, str):
            if hasattr(freq, "freqstr"):
                freq = freq.freqstr
            else:
                raise ValueError(f"Invalid freq {freq}")

        match = maybe.expect(
            re.match(r"(?P<n>\d+)?\s*(?P<freq>(?:\w|\-)+)", freq),
            f"Unsupported freq format {freq}",
        )
        groups = match.groupdict()

        n = maybe.map_or(groups["n"], int, 1)
        freq = groups["freq"].upper().split("-")[0]

        return cls(cls._freq_pandas_to_numpy[freq], n)

    def to_pandas(self) -> str:
        if self.np_freq == ("M", 3):
            return f"{self.multiple}Q"

        np_name, np_multiple = self.np_freq

        return _canonical_freqstr(
            self.multiple, self._freq_numpy_to_pandas[np_name]
        )


@dataclass
class _BasePeriod:
    data: np.datetime64
    multiple: int

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
    def freq(self):
        freq = Freq(np.datetime_data(self.data.dtype), self.multiple)
        return freq.to_pandas()

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
                self.data + self.multiple * other, multiple=self.multiple
            )

    def __sub__(self, other):
        if _is_number(other):
            return self.__class__(
                self.data - self.multiple * other, multiple=self.multiple
            )
        else:
            return self.data - other.data

        raise ValueError(other)


@functools.total_ordering
class Period(_BasePeriod):
    def periods(self, count: int):
        return _periods(self.data, count, self.multiple)

    def to_pandas(self):
        import pandas as pd

        return pd.Period(self.data, self.freq)

    def to_timestamp(self):
        return self.data.astype(object)

    def __repr__(self):
        return f"Period<{self.data}, {self.freq}>"

    def __lt__(self, other: Period):
        return self.data < other.data


def _periods(start: np.datetime64, count: int, multiple: int) -> Periods:
    return Periods(np.arange(start, multiple * count, multiple), multiple)


class Periods(_BasePeriod):
    def first(self) -> Period:
        return self[0]

    def last(self) -> Period:
        return self[-1]

    def take(self, count: int) -> Periods:
        return self[:count]

    def tail(self, count: int) -> Periods:
        return self[-count:]

    def future(self, count: int) -> Periods:
        return _periods(self.data[-1] + self.multiple, count, self.multiple)

    def past(self, count: int) -> Periods:
        return _periods(self.data[0] - count, count, self.multiple)

    def extend(self, count: int) -> Periods:
        return Periods(
            np.concatenate([self.data, self.future(count).data]),
            multiple=self.multiple,
        )

    def to_pandas(self):
        import pandas as pd

        # older versions of pandas expect ns-datetime64
        return pd.PeriodIndex(self.data.astype("M8[ns]"), freq=self.freq)

    def intersection(self, other):
        return self.data[np.in1d(self, other)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Periods:
        if _is_number(idx):
            return Period(self.data[idx], multiple=self.multiple)

        return Periods(self.data[idx], multiple=self.multiple)

    def __eq__(self, other):
        if not isinstance(other, Periods):
            return False

        return self.multiple == other.multiple and np.array_equal(
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

    return Period(
        np.datetime64(data, freq.np_freq),
        multiple=freq.multiple,
    )


def periods(start, freq, count: int) -> Period:
    return period(start, freq).periods(count)
