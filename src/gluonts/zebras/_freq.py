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

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np

from gluonts import maybe
from gluonts.core import serde

NpFreq = Tuple[str, int]


def _canonical_freqstr(n: int, name: str) -> str:
    """Canonical name of frequency.

    >>> _canonical_freqstr("X")
    'X'
    >>> _canonical_freqstr("3X")
    '3X'

    This allows us to easily string compare frequencies
    (solves ``"1X" != "X"``).
    """

    if n == 1:
        return name

    return f"{n}{name}"


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
    {
        pd_freq: (np_freq, 1)
        for np_freq, pd_freq in _freq_numpy_to_pandas.items()
    },
    **{
        "A": ("Y", 1),
        "AS": ("Y", 1),
        "YS": ("Y", 1),
        "MS": ("M", 1),
        "MIN": ("m", 1),
        "Q": ("M", 3),
        "QS": ("M", 3),
        "B": ("D", 1),
        "W": ("D", 1),
    },
)


@dataclass
class Freq:
    """
    A class representing frequencies, such as n-days.

    Note: Use ``freq`` to construct instances of ``Freq``.

    We use frequency aliases from pandas over frequency names defined by numpy.
    For example, the name for minutely is either "min" or "T", while "m"
    and "M" represent monthly frequencies. In contrast numpy uses "m" for
    minutely and "M" for monthly. In addition, pandas defines some frequencies
    which do not exist in numpy, for example quarterly frequencies.

    However, internally we use ``numpy.datetime64`` objects and thus we must
    support numpy's frequency names as well. To do this we generally use base
    frequencies (multiple = 1), since numpy otherwise aligns timestamps for us
    which we don't want.

    Weekly frequency needs to be handled specially, since numpy counts the
    number of weeks since Thu Jan 1 1970 and uses Thursday and aligns the
    timestamp to Thursday. We therefore use daily frequency internally and
    align to Monday.
    """

    name: str
    n: int

    @property
    def np_freq(self) -> NpFreq:
        return _freq_pandas_to_numpy[self.name]

    @classmethod
    def __get_validators__(cls):
        # pydantic support
        yield freq

    @classmethod
    def from_pandas(cls, freq) -> Freq:
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

        name = groups["freq"].upper().split("-")[0]
        n = maybe.map_or(groups["n"], int, 1)

        return cls(name, n)

    def to_pandas(self) -> "pandas.Period":
        from pandas.tseries.frequencies import to_offset

        return to_offset(str(self))

    def shift(self, start: np.datetime64, count: int) -> np.datetime64:
        if self.name == "B":
            return np.busday_offset(data, self.n * count)

        return data + self.n * count

    def range(self, start: np.datetime64, count: int) -> np.datetime64:
        if self.name == "B":
            # We first collect all days, even non business days to then filter
            # for business days, of which we then take, each n-th.
            periods = np.arange(start, np.busday_offset(start, count * self.n))
            periods = periods[np.is_busday(periods)]
            return periods[:: self.n]

        step = self.n

        if self.name == "W":
            step *= 7

        return np.arange(start, count * step, step)

    def __str__(self) -> str:
        return _canonical_freqstr(self.n, self.name)


@serde.encode.register
def _encode_freq(v: Freq) -> dict:
    return {
        "__kind__": "instance",
        "class": "gluonts.zebras.Freq.from_pandas",
        "args": [v.to_pandas()],
    }


def freq(arg) -> Freq:
    if isinstance(arg, Freq):
        return arg

    return Freq.from_pandas(arg)
