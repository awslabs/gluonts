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
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from gluonts import maybe

NpFreq = Tuple[str, int]


def _canonical_freqstr(n: int, name: str, suffix: Optional[str] = None) -> str:
    """Canonical name of frequency.

    >>> _canonical_freqstr(1, "X")
    'X'
    >>> _canonical_freqstr(3, "X")
    '3X'
    >>> _canonical_freqstr(3, "X", "Y")
    '3X-Y'

    This allows us to easily string compare frequencies
    (solves ``"1X" != "X"``).
    """

    if suffix:
        name = f"{name}-{suffix}"

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


def _canonical_name(name: str) -> str:
    return {"MIN": "T", "Y": "A"}.get(name, name)


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
    suffix: Optional[str] = None

    def __post_init__(self):
        self.name = _canonical_name(self.name)

    @property
    def __init_passed_kwargs__(self):
        return {"name": self.name, "n": self.n}

    @property
    def np_freq(self) -> NpFreq:
        return _freq_pandas_to_numpy[self.name]

    @property
    def step(self):
        if self.name == "W":
            return self.n * 7

        return self.n

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
                raise ValueError(f"Invalid freq {freq}: {type(freq)}")

        match = maybe.expect(
            re.match(
                r"(?P<n>\d+)?\s*(?P<freq>\w+)(?P<suffix>\-\w+)?",
                freq.upper(),
            ),
            f"Unsupported freq format {freq}",
        )

        groups = match.groupdict()
        n = maybe.map_or(groups["n"], int, 1)

        suffix = groups["suffix"]
        if suffix is not None:
            # remove leading `-` from `-MON`
            suffix = suffix[1:]

        return cls(groups["freq"], n, suffix)

    def to_pandas(self):
        from pandas.tseries.frequencies import to_offset

        return to_offset(str(self))

    def shift(self, start: np.datetime64, count: int) -> np.datetime64:
        if self.name == "B":
            return np.busday_offset(start, self.n * count)

        return start + self.step * count

    def range(self, start: np.datetime64, count: int) -> np.ndarray:
        if self.name == "B":
            # We first collect all days, even non business days to then filter
            # for business days, of which we then take, each n-th.
            periods = np.arange(start, np.busday_offset(start, count * self.n))
            periods = periods[np.is_busday(periods)]
            return periods[:: self.n]

        return np.arange(start, count * self.step, self.step)

    def __str__(self) -> str:
        return _canonical_freqstr(self.n, self.name, self.suffix)


def freq(arg) -> Freq:
    if isinstance(arg, Freq):
        return arg

    return Freq.from_pandas(arg)
