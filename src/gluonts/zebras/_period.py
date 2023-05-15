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
from dataclasses import dataclass
from typing import Any, Union, Optional, cast, overload

import numpy as np
from dateutil.parser import parse as du_parse

from gluonts.core import serde
from gluonts import maybe

from ._freq import Freq


weekday_offsets = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}


def _is_number(value):
    return isinstance(value, (int, np.integer))


class _BasePeriod:
    data: Any
    freq: Freq

    @property
    def freqstr(self) -> str:
        return str(self.freq)

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

    def unix_epoch(self) -> int:
        return self.to_numpy().astype("M8[s]").astype(int)

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

    @classmethod
    def from_pandas(cls, index):
        """Turn ``pandas.PeriodIndex`` or ``pandas.DatetimeIndex`` into
        ``Periods``.
        """

        import pandas as pd

        if isinstance(index, pd.DatetimeIndex):
            index = index.to_period()
        else:
            assert isinstance(index, pd.PeriodIndex)

        freq = Freq.from_pandas(index.freqstr)
        np_index = np.array(index.asi8, dtype=f"M8[{freq.np_freq[0]}]")
        assert np.all(np.diff(np_index).astype(int) == freq.n)

        return Periods(np_index, freq)

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
        assert 0 <= idx <= len(self), idx

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

        return len(self) == len(other) and self.start == other.start

    def to_numpy(self) -> np.ndarray:
        return self.data

    def __array__(self) -> np.ndarray:
        return self.data

    def unix_epoch(self) -> np.ndarray:
        return self.to_numpy().astype("M8[s]").astype(int)


@serde.encode.register
def _encode_zebras_periods(v: Periods):
    return {
        "__kind__": "instance",
        "class": "gluonts.zebras.periods",
        "kwargs": serde.encode(
            {"start": v.start, "freq": str(v.freq), "count": len(v)}
        ),
    }


def period(
    data: Union[Period, str], freq: Optional[Union[Freq, str]] = None
) -> Period:
    """Create a ``zebras.Period`` object that represents a period of time.

    Parameters
    ----------
    data
        The time period represented by a string (e.g., "2023-01-01"),
        or another Period object.
    freq, optional
        The frequency of the period, e.g, "H" for hourly, by default None.

    Returns
    -------
        A ``zebras.Period`` object.
    """
    if freq is None:
        if hasattr(data, "freqstr"):
            freq = Freq.from_pandas(data.freqstr)
        else:
            raise ValueError("No frequency specified.")
    elif isinstance(freq, Freq):
        freq = freq
    elif isinstance(freq, str):
        freq = Freq.from_pandas(freq)
    else:
        raise ValueError(f"Unknown frequency type {type(freq)}.")

    data_: Any

    if isinstance(data, Period):
        data_ = data.data

    elif isinstance(data, str):
        data_ = du_parse(
            data,
            default=datetime.datetime(1970, 1, 1),
            ignoretz=True,
        )
    else:
        # TODO: should we add a check?
        data_ = data

    if freq.name == "W":
        period = Period(np.datetime64(data_, freq.np_freq), freq)
        weekday_offset = maybe.map_or(
            freq.suffix, weekday_offsets.__getitem__, 0
        )
        period.data -= (cast(int, period.dayofweek) - weekday_offset) % 7
        return period

    return Period(np.datetime64(data_, freq.np_freq), freq)


def periods(
    start: Union[Period, str], freq: Union[Freq, str], count: int
) -> Period:
    """Create a ``zebras.Periods`` object that represents multiple consecutive
    periods of time.

    Parameters
    ----------
    start
        The starting time period represented by a string (e.g., "2023-01-01"),
        or another Period object.
    freq
        The frequency of the period, e.g, "H" for hourly.
    count
        The number of periods.

    Returns
    -------
        A ``zebras.Periods`` object.
    """
    return period(start, freq).periods(count)
