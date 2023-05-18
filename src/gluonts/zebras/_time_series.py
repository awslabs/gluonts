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
from operator import itemgetter
from typing import Collection, Union, Optional, Dict, List

import numpy as np
from toolz import first

from gluonts import maybe
from gluonts.itertools import pluck_attr

from ._base import Pad, TimeBase
from ._period import period, Period, Periods
from ._util import AxisView, pad_axis, _replace


@dataclasses.dataclass(eq=False)
class TimeSeries(TimeBase):
    values: np.ndarray
    index: Optional[Periods] = None
    name: Optional[str] = None
    tdim: int = -1
    metadata: Optional[dict] = None
    _pad: Pad = Pad()

    def __post_init__(self):
        assert maybe.map_or(self.index, len, len(self)) == len(self), (
            "Index has incorrect length. "
            f"Expected: {len(self)}, got {len(self.index)}."
        )

    def __eq__(self, other):
        return self.values == other

    def __array__(self):
        return self.values

    def to_numpy(self):
        return self.values

    def __len__(self):
        return self.values.shape[self.tdim]

    def _slice_tdim(self, idx):
        if isinstance(idx, int):
            return AxisView(self.values, self.tdim)[idx]

        start, stop, step = idx.indices(len(self))
        assert step == 1

        return _replace(
            self,
            values=AxisView(self.values, self.tdim)[idx],
            index=maybe.map(self.index, itemgetter(idx)),
            _pad=self._pad.extend(-start, stop - len(self)),
        )

    def pad(self, value, left: int = 0, right: int = 0) -> TimeSeries:
        assert left >= 0 and right >= 0

        values = pad_axis(
            self.values,
            axis=self.tdim,
            left=left,
            right=right,
            value=value,
        )

        index = self.index
        if self.index is not None:
            index = self.index.prepend(left).extend(right)

        return _replace(
            self,
            values=values,
            index=index,
            _pad=self._pad.extend(left, right),
        )

    @staticmethod
    def _batch(xs: List[TimeSeries]) -> BatchTimeSeries:
        for series in xs:
            assert type(series) == TimeSeries

        pluck = pluck_attr(xs)

        tdims = set(pluck("tdim"))
        assert len(tdims) == 1
        tdim = first(tdims)
        if tdim >= 0:
            # We insert a new axis at the front, so if tdim is counting from
            # the left (tdim is positive) we need to shift by one to the right.
            tdim += 1

        values = np.stack(pluck("values"))

        return BatchTimeSeries(
            values=values,
            tdim=tdim,
            index=pluck("index"),
            name=pluck("name"),
            metadata=pluck("metadata"),
            _pad=pluck("_pad"),
        )

    def plot(self):
        import matplotlib.pyplot as plt

        if self.index is None:
            plt.plot(self.values)
        else:
            plt.plot(self.index, self.values)


@dataclasses.dataclass
class BatchTimeSeries(TimeBase):
    values: np.ndarray
    index: List[Optional[Periods]]
    name: List[Optional[str]]
    tdim: int
    metadata: List[Optional[dict]]
    _pad: List[Pad]

    def _slice_tdim(self, idx):
        if isinstance(idx, int):
            return AxisView(self.values, self.tdim)[idx]

        start, stop, step = idx.indices(len(self))
        assert step == 1

        def calc_pad(pad):
            pad_left = max(0, pad.left - start)
            pad_right = max(0, pad.right + stop + 1 - len(self) - pad_left)

            return Pad(pad_left, pad_right)

        return _replace(
            self,
            values=AxisView(self.values, self.tdim)[idx],
            index=[maybe.map(index, itemgetter(idx)) for index in self.index],
            _pad=list(map(calc_pad, self._pad)),
        )

    @property
    def batch_size(self):
        return len(self.values)

    def __len__(self):
        return self.values.shape[self.tdim]

    def __array__(self):
        return self.values

    def items(self):
        return TimeSeriesItems(self)

    def pad(self, value, left: int = 0, right: int = 0) -> TimeSeries:
        assert left >= 0 and right >= 0

        values = pad_axis(
            self.values,
            axis=self.tdim,
            left=left,
            right=right,
            value=value,
        )

        def extend_index(index):
            return index.prepend(left).extend(right)

        return _replace(
            self,
            values=values,
            index=[maybe.map(index_, extend_index) for index_ in self.index],
            _pad=[pad.extend(left, right) for pad in self._pad],
        )

    def like(self, values: np.ndarray, name: Optional[str] = None):
        return _replace(self, values=values, name=name)


@dataclasses.dataclass(repr=False)
class TimeSeriesItems:
    data: BatchTimeSeries

    def __len__(self):
        return self.data.batch_size

    def __getitem__(self, idx):
        tdim = self.data.tdim

        if isinstance(idx, int):
            cls = TimeSeries
            if tdim > 0:
                tdim -= 1
        else:
            cls = BatchTimeSeries

        return cls(
            values=self.data.values[idx],
            index=self.data.index[idx],
            name=self.data.name[idx],
            metadata=self.data.metadata[idx],
            _pad=self.data._pad[idx],
            tdim=tdim,
        )


def time_series(
    values: Collection,
    *,
    index: Optional[Periods] = None,
    start: Optional[Union[Period, str]] = None,
    freq: Optional[str] = None,
    tdim: int = -1,
    name: Optional[str] = None,
    metadata: Optional[Dict] = None,
):
    """Create a ``zebras.TimeSeries`` object that represents a time series.

    Parameters
    ----------
    values
        A sequence (e.g., list, numpy arrays) representing the values
        of the time series.
    index, optional
        A ``zebras.Periods`` object representing timestamps.
        Must have the same length as the `values`, by default None
    start, optional
        The start time represented by a string (e.g., "2023-01-01"),
        or a ``zebras.Period`` object. An index will be constructed using
        this start time and the specificed frequency, by default None
    freq, optional
        The frequency of the period, e.g, "H" for hourly, by default None
    tdim, optional
        The time dimension in `values`, by default -1
    name: optional
        A description for the time series. This will be the column names when
        returned from a ``TimeFrame``.
    metadata, optional
        A dictionary of metadata associated with the time series, by default None

    Returns
    -------
        A ``zebras.TimeSeries`` object.
    """
    values = np.array(values)

    ts = TimeSeries(
        values,
        index=index,
        tdim=tdim,
        name=name,
        metadata=metadata,
    )

    if ts.index is None and start is not None:
        if freq is not None:
            start = period(start, freq)
        else:
            assert isinstance(start, Period)

        return ts.with_index(start.periods(len(ts)))

    return ts
