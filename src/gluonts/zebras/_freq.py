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

import re
from dataclasses import dataclass, asdict
from typing import Tuple

from gluonts import maybe
from gluonts.core import serde

NpFreq = Tuple[str, int]


def _canonical_freqstr(n: int, freq: str):
    if n == 1:
        return freq

    return f"{n}{freq}"


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
    np_freq: NpFreq
    pd_freq: str
    _multiple: int

    @property
    def multiple(self):
        if self.pd_freq == "W":
            return self._multiple * 7

        return self._multiple

    @classmethod
    def __get_validators__(cls):
        yield freq

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

        return cls(_freq_pandas_to_numpy[freq], freq, n)

    def to_pandas(self) -> str:
        return _canonical_freqstr(self._multiple, self.pd_freq)

    def __str__(self):
        return self.to_pandas()


@serde.encode.register
def _encode_freq(v: Freq):
    return {
        "__kind__": "instance",
        "class": "gluonts.zebras.Freq.from_pandas",
        "args": [v.to_pandas()],
    }


def freq(arg) -> Freq:
    if isinstance(arg, Freq):
        return arg

    return Freq.from_pandas(arg)
