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

import dataclasses
from typing import Any, Optional, NamedTuple, Union
from typing_extensions import Literal


from gluonts import maybe

from ._period import Periods, Period
from ._util import _replace


LeftOrRight = Literal["l", "r"]


class Pad(NamedTuple):
    """Indicator for padded values.

    >>> from gluonts.zebras import time_series
    >>> ts = time_series([1, 2, 3]).pad(0, left=2, right=2)
    >>> assert len(ts == 7)
    >>> assert list(ts) == [0, 0, 1, 2, 3, 0, 0]
    >>> assert ts._pad.left == 2
    >>> assert ts._pad.right == 2

    """

    left: int = 0
    right: int = 0

    def extend(self, left, right):
        return Pad(
            left=max(0, self.left + left),
            right=max(0, self.right + right),
        )


@dataclasses.dataclass
class ILoc:
    tb: TimeBase

    def __getitem__(self, idx):
        return self.tb._slice_tdim(idx)


@dataclasses.dataclass
class TimeView:
    tf: TimeBase

    def __getitem__(self, idx):
        assert isinstance(idx, slice)
        assert idx.step is None or idx.step == 1

        start = maybe.map(idx.start, self.tf.index_of)
        stop = maybe.map(idx.stop, self.tf.index_of)

        return ILoc(self.tf)[start:stop]


class TimeBase:
    index: Any

    def _slice_tdim(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def pad(self, value, left: int = 0, right: int = 0) -> TimeBase:
        raise NotImplementedError

    @property
    def iloc(self):
        return ILoc(self)

    @property
    def loc(self):
        return TimeView(self)

    @property
    def start(self):
        return self.iloc[0]

    @property
    def end(self):
        return self.iloc[-1]

    def head(self, count: int) -> Periods:
        return self.iloc[:count]

    def tail(self, count: int) -> Periods:
        if count is None:
            return self

        return self.iloc[-count:]

    def resize(
        self,
        length: Optional[int],
        pad_value=0,
        pad: LeftOrRight = "l",
        skip: LeftOrRight = "r",
    ) -> TimeBase:
        """Force time frame to have length ``length``.

        This pads or slices the time frame, depending on whether its size is
        smaller or bigger than the required length.

        By default we pad values on the left, and skip on the right.
        """
        assert pad in ("l", "r")
        assert skip in ("l", "r")

        if length is None or len(self) == length:
            return self

        if len(self) < length:
            left = right = 0
            if pad == "l":
                left = length - len(self)
            else:
                right = length - len(self)

            return self.pad(pad_value, left=left, right=right)

        if skip == "l":
            return self.iloc[len(self) - length :]
        else:
            return self.iloc[: length - len(self)]

    def index_of(self, period: Union[Period, str]) -> int:
        assert self.index is not None

        return self.index.index_of(period)

    def __getitem__(self, idx: Union[slice, int, str]):
        subtype: Optional[int]

        if isinstance(idx, int):
            subtype = idx
        else:
            assert isinstance(idx, slice)

            start = idx.start
            if start is not None or not isinstance(start, int):
                if isinstance(start, (Period, str)):
                    start = self.index_of(start)

            stop = idx.stop
            if stop is not None or not isinstance(stop, int):
                if isinstance(stop, (Period, str)):
                    stop = self.index_of(stop)

            idx = slice(start, stop, idx.step)

        return self.iloc[idx]

        raise RuntimeError(f"Unsupported type {subtype}.")

    def with_index(self, index):
        return _replace(self, index=index)
