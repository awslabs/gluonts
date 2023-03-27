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

import copy
import dataclasses
from operator import itemgetter
from typing import Optional, List, NamedTuple, Union, Collection, Dict, Any
from typing_extensions import Literal

import numpy as np
from toolz import first, keymap, valmap, dissoc, merge, itemmap

from gluonts import maybe
from gluonts.itertools import pluck_attr, columns_to_rows, select

from ._period import Periods, Period, period
from ._repr import html_table
from ._util import AxisView, pad_axis, _replace

LeftOrRight = Literal["l", "r"]


class Pad(NamedTuple):
    """Indicator for padded values.

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
    tf: TimeFrame

    def __getitem__(self, idx):
        assert isinstance(idx, slice)
        assert idx.step is None or idx.step == 1

        start = maybe.map(idx.start, self.tf.index_of)
        stop = maybe.map(idx.stop, self.tf.index_of)

        return ILoc(self.tf)[start:stop]


class TimeBase:
    def _slice_tdim(self, idx):
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
    ) -> TimeFrame:
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

    def index_of(self, period: Period) -> int:
        assert self.index is not None

        return self.index.index_of(period)

    def __getitem__(self, idx: Union[slice, int, str]):
        if isinstance(idx, int):
            subtype = idx
        else:
            assert isinstance(idx, slice)
            subtype = maybe.or_(idx.start, idx.stop)

        if subtype is None:
            return self

        elif isinstance(subtype, int):
            return self.iloc[idx]

        elif isinstance(subtype, Period):
            return self.loc[idx]

        raise RuntimeError(f"Unsupported type {subtype}.")

    def with_index(self, index):
        return _replace(self, index=index)


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

    @classmethod
    def batch(cls, xs: List[TimeSeries], into=None):
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
        if into is not None:
            values = into(values)

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

    @property
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


@dataclasses.dataclass
class TimeFrame(TimeBase):
    columns: dict
    index: Optional[Periods]
    static: dict
    length: int
    tdims: dict
    metadata: Optional[dict] = None
    default_tdim: int = -1
    _pad: Pad = Pad()

    def __post_init__(self):
        for column in self.columns:
            self.tdims.setdefault(column, self.default_tdim)

        for column in self.columns:
            assert len(self._time_view(column)) == self.length, (
                f"Column {column!r} has incorrect length in time dimension. "
                f"Expected: {len(self)}, got {len(self._time_view(column))}."
            )

        assert maybe.map_or(self.index, len, self.length) == self.length, (
            f"Index has incorrect length. "
            f"Expected: {len(self)}, got {len(self.index)}."
        )

    def _time_view(self, column):
        """View of column with respect to time."""

        return AxisView(self.columns[column], self.tdims[column])

    def _slice_tdim(self, idx):
        start, stop, step = idx.indices(len(self))
        assert step == 1

        pad_left = max(0, self._pad.left - start)
        pad_right = max(0, self._pad.right + stop + 1 - len(self) - pad_left)

        return _replace(
            self,
            columns={
                column: self._time_view(column)[idx] for column in self.columns
            },
            index=maybe.map(self.index, itemgetter(idx)),
            length=stop - start,
            _pad=Pad(pad_left, pad_right),
        )

    def __getitem__(self, idx: Union[slice, int, str]):
        if isinstance(idx, slice):
            subtype = maybe.or_(idx.start, idx.stop)
        else:
            subtype = None

        if isinstance(idx, int) or isinstance(subtype, int):
            return self.iloc[idx]

        return TimeSeries(
            self.columns[idx],
            index=self.index,
            tdim=self.tdims[idx],
            metadata=self.metadata,
            name=idx,
            _pad=self._pad,
        )

    def pad(self, value, left: int = 0, right: int = 0) -> TimeFrame:
        assert left >= 0 and right >= 0

        columns = {
            column: pad_axis(
                self.columns[column],
                axis=self.tdims[column],
                left=left,
                right=right,
                value=value,
            )
            for column in self.columns
        }
        length = self.length + left + right
        pad_left = left + self._pad.left
        pad_right = right + self._pad.right

        index = self.index
        if self.index is not None:
            index = self.index.prepend(left).extend(right)

        return _replace(
            self,
            columns=columns,
            index=index,
            length=length,
            _pad=Pad(pad_left, pad_right),
        )

    def astype(self, type, columns=None) -> TimeFrame:
        if columns is None:
            columns = self.columns

        return _replace(
            self,
            columns=valmap(
                lambda col: col.astype(type), select(columns, self.columns)
            ),
        )

    def __repr__(self) -> str:
        columns = ", ".join(self.columns)
        return f"TimeFrame<size={len(self)}, columns=[{columns}]>"

    def _table_columns(self):
        columns = {}

        if self.index is not None:
            index = pluck_attr(self.index, "data")
            if len(self) > 10:
                index = [*index[:5], "...", *index[-5:]]

            columns[""] = index

        def move_axis(data, name):
            return np.moveaxis(data, self.tdims[name], 0)

        if len(self) > 10:
            head = self.head(5)
            tail = self.tail(5)

            columns.update(
                {
                    col: [
                        *(move_axis(head[col], col)),
                        "...",
                        *(move_axis(tail[col], col)),
                    ]
                    for col in self.columns
                }
            )
        else:
            columns.update(
                {
                    name: move_axis(values, name)
                    for name, values in self.columns.items()
                }
            )

        return columns

    def _repr_html_(self):
        columns = self._table_columns()

        html = [
            html_table(columns),
            f"{len(self)} rows × {len(self.columns)} columns",
        ]

        if self.static:
            html.extend(
                [
                    "<h3>Static Data</h3>",
                    html_table(
                        {name: [val] for name, val in self.static.items()}
                    ),
                ]
            )

        return "\n".join(html)

    def set(self, name, value, tdim=None):
        assert name not in self.static

        tdim = maybe.unwrap_or(tdim, self.default_tdim)

        return _replace(
            self,
            columns=merge(self.columns, {name: value}),
            tdims=merge(self.tdims, {name: tdim}),
        )

    def set_static(self, name, value):
        assert name not in self.columns

        return _replace(self, static=merge(self.static, {name: value}))

    def set_like(self, ref: str, column, value, tdim=None):
        assert ref in self.columns

        return self.set(column, value, tdim)

    def remove(self, column):
        return _replace(
            self,
            columns=dissoc(self.columns, column),
            tdims=dissoc(self.tdims, column),
        )

    def remove_static(self, name):
        return _replace(self, static=dissoc(self.static, name))

    def like(self, columns=None, static=None):
        columns = maybe.unwrap_or(columns, {})
        static = maybe.unwrap_or(static, {})

        return _replace(self, columns=columns, static=static)

    def stack(
        self,
        select: List[str],
        into: str,
        drop: bool = True,
    ) -> TimeFrame:
        # Ensure all tdims are the same.
        # TODO: Can we make that work for different tdims? There might be a
        # problem with what the resulting dimensions are.
        all_tdims = set(self.tdims[column] for column in select)
        assert len(all_tdims) == 1
        tdim = first(all_tdims)

        if drop:
            columns = dissoc(self.columns, *select)
            tdims = dissoc(self.tdims, *select)
        else:
            columns = dict(self.columns)
            tdims = dict(self.tdims)

        columns[into] = np.vstack([self.columns[column] for column in select])
        tdims[into] = tdim

        return _replace(self, columns=columns, tdims=tdims)

    def as_dict(self, prefix=None, static=True):
        result = dict(self.columns)

        if prefix is not None:
            result = {prefix + key: value for key, value in result.items()}

        if static:
            result.update(self.static)

        return result

    def split(
        self,
        index,
        past_length=None,
        future_length=None,
        pad_value=0.0,
    ):
        if not isinstance(index, (int, np.integer)):
            index = self.index_of(index)
        elif index < 0:
            index = len(self) + index

        if past_length is None:
            past_length = index

        if future_length is None:
            future_length = len(self) - index

        if self.index is None:
            new_index = None
        else:
            new_index = (
                self.index.start + (len(self) - index - past_length)
            ).periods(past_length + future_length)

        pad_left = max(0, past_length - index)
        pad_right = max(0, future_length - (len(self) - index))
        self = self.pad(pad_value, pad_left, pad_right)

        index += pad_left

        def split_item(item):
            name, data = item

            tdim = self.tdims[name]
            past, future = np.split(data, [index], axis=tdim)
            past = AxisView(past, tdim)[-past_length:]
            future = AxisView(future, tdim)[:future_length]
            return name, (past, future)

        past, future = columns_to_rows(itemmap(split_item, self.columns))

        # past_index = self.index[:past_length]
        # future_index = self.index[past_length:]
        # new_index = past_index.prepend(pad_left).tail(past_length).extend(future_length)
        # future_index.extend(pad_right).head(future_length)

        return SplitFrame(
            _past=past,
            _future=future,
            index=new_index,
            static=self.static,
            past_length=past_length,
            future_length=future_length,
            tdims=self.tdims,
            metadata=self.metadata,
            _pad=self._pad,
        )

    def apply(self, fn, columns=None):
        if columns is None:
            columns = self.columns.keys()

        return _replace(self, columns=valmap(fn, self.columns))

    def __len__(self) -> int:
        return self.length


@dataclasses.dataclass
class SplitFrame:
    _past: dict
    _future: dict
    index: Optional[Periods]
    static: dict
    past_length: int
    future_length: int
    tdims: dict
    metadata: Optional[dict] = None
    default_tdim: int = -1
    _pad: Pad = Pad()

    @property
    def past(self):
        return TimeFrame(
            self._past,
            index=maybe.map(
                self.index, lambda index: index[: self.past_length]
            ),
            static=self.static,
            length=self.past_length,
            tdims=self.tdims,
            metadata=self.metadata,
            _pad=Pad(self._pad.left, 0),
        )

    @property
    def future(self):
        return TimeFrame(
            self._future,
            index=maybe.map(
                self.index, lambda index: index[-self.future_length :]
            ),
            static=self.static,
            length=self.future_length,
            tdims=self.tdims,
            metadata=self.metadata,
            _pad=Pad(0, self._pad.right),
        )

    def __len__(self):
        return self.past_length + self.future_length

    def set(self, name, value, tdim=None):
        tdim = maybe.unwrap_or(tdim, self.default_tdim)
        assert value.shape[tdim] == len(self)

        past, future = np.split(
            value,
            [self.past_length],
            axis=tdim,
        )

        return _replace(
            past=merge(self.past, {name: past}),
            future=merge(self.future, {name: future}),
            tdims=merge(self.tdims, {name: tdim}),
        )

    def set_like(self, ref: str, column, value, tdim=None):
        is_past = ref in self._past
        is_future = ref in self._future

        if is_past:
            if is_future:
                return self.set(column, value, tdim)
            else:
                return self.set_past(column, value, tdim)
        elif is_future:
            return self.set_future(column, value, tdim)

        raise KeyError(f"Ref {ref} is neither past nor future")

    def set_past(self, name, value, tdim=None):
        tdim = maybe.unwrap_or(tdim, self.default_tdim)
        assert value.shape[tdim] == self.past_length
        assert self.tdims.get(name, tdim) == tdim

        return _replace(
            past=merge(self.past, {name: value}),
            tdims=merge(self.tdims, {name: tdim}),
        )

    def set_future(self, name, value, tdim=None):
        tdim = maybe.unwrap_or(tdim, self.default_tdim)
        assert value.shape[tdim] == self.future_length
        assert self.tdims.get(name, tdim) == tdim

        return _replace(
            future=merge(self.future, {name: value}),
            tdims=merge(self.tdims, {name: tdim}),
        )

    def remove(self, column):
        return _replace(
            self,
            columns=dissoc(self.columns, column),
            tdims=dissoc(self.tdims, column),
        )

    def _repr_html_(self):
        past = self.past._table_columns()
        future = self.future._table_columns()

        length = max(
            len(first(past.values())) if past else 0,
            len(first(future.values())) if future else 0,
        )

        def pad(col):
            to_pad = length - len(col)

            return list(col) + [""] * to_pad

        past = valmap(pad, past)
        future = valmap(pad, future)

        past = keymap(lambda key: f"past_{key}" if key else "past", past)
        future = keymap(
            lambda key: f"future_{key}" if key else "future", future
        )

        return html_table({**past, "|": ["|"] * length, **future})

    def as_dict(self):
        past = keymap(lambda key: f"past_{key}", self._past)
        future = keymap(lambda key: f"future_{key}", self._future)

        return {**past, **future, **self.static}

    def resize(
        self,
        past_length: Optional[int] = None,
        future_length: Optional[int] = None,
        pad_value=0.0,
    ) -> SplitFrame:
        index = self.index

        past_length = maybe.unwrap_or(past_length, self.past_length)
        future_length = maybe.unwrap_or(future_length, self.future_length)

        if index is not None:
            start = index[0] + (self.past_length + past_length)
            index = start.periods(past_length + future_length)

        return _replace(
            self,
            _past=self.past.resize(
                past_length, pad_value, pad="l", skip="l"
            ).columns,
            past_length=maybe.unwrap_or(past_length, self.past_length),
            _future=self.future.resize(
                future_length, pad_value, pad="r", skip="r"
            ).columns,
            future_length=maybe.unwrap_or(future_length, self.future_length),
            index=index,
        )

    def with_index(self, index):
        return _replace(self, index=index)


def time_frame(
    columns: Optional[Dict[str, Collection]] = None,
    *,
    index: Optional[Periods] = None,
    start: Optional[Union[Period, str]] = None,
    freq: Optional[str] = None,
    static: Optional[Dict[str, Any]] = None,
    tdims: Optional[Dict[str, int]] = None,
    length: Optional[int] = None,
    default_tdim: int = -1,
    metadata: Optional[Dict] = None,
):
    """Create a ``zebras.TimeFrame`` object that represents one
    or more time series.

    Parameters
    ----------
    columns, optional
        A dictionary where keys are strings representing column names and
        values are sequences (e.g., list, numpy arrays). All columns must have
        the same length in the time dimension, by default None
    index, optional
        A ``zebras.Periods`` object representing timestamps.
        Must have the same length as the columns, by default None
    start, optional
        The start time represented by a string (e.g., "2023-01-01"),
        or a ``zebras.Period`` object. An index will be constructed using
        this start time and the specificed frequency, by default None
    freq, optional
        The frequency of the period, e.g, "H" for hourly, by default None
    static, optional
        A dictionary of static-in-time features, by default None
    tdims, optional
        A dictionary specifying the time dimension for each column. The keys
        should match those in `columns`. If unspecified for a column,
        the `default_tdim` is used, by default None
    length, optional
        The length (in time) of the TimeFrame, by default None
    default_tdim, optional
        The default time dimension, by default -1
    metadata, optional
        A dictionary of metadata associated with the TimeFrame, by default None

    Returns
    -------
        A ``zebras.TimeFrame`` object.
    """
    assert (
        index is None or start is None
    ), "Both index and start cannot be specified."

    columns = maybe.unwrap_or_else(columns, dict)
    tdims = maybe.unwrap_or_else(tdims, dict)
    static = maybe.unwrap_or_else(static, dict)

    columns = valmap(np.array, columns)
    static = valmap(np.array, static)

    if length is None:
        if index is not None:
            length = len(index)
        elif columns:
            column = first(columns)
            length = columns[column].shape[tdims.get(column, default_tdim)]
        else:
            length = 0

    tf = TimeFrame(
        columns=columns,
        index=index,
        static=static,
        tdims=tdims,
        length=length,
        default_tdim=default_tdim,
        metadata=metadata,
    )
    if tf.index is None and start is not None:
        if freq is not None:
            start = period(start, freq)

        return tf.with_index(start.periods(len(tf)))

    return tf


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

        return ts.with_index(start.periods(len(ts)))

    return ts


def split_frame(
    full=None,
    *,
    past=None,
    future=None,
    past_length=None,
    future_length=None,
    metadata=None,
    static=None,
    start=None,
    freq=None,
    index=None,
    tdims=None,
    default_tdim=-1,
):
    full = valmap(np.array, maybe.unwrap_or_else(full, dict))
    past = valmap(np.array, maybe.unwrap_or_else(past, dict))
    future = valmap(np.array, maybe.unwrap_or_else(future, dict))
    static = valmap(np.array, maybe.unwrap_or_else(static, dict))
    tdims = maybe.unwrap_or_else(tdims, dict)

    full_length = None

    if full:
        column = first(full)
        full_length = full[column].shape[tdims.get(column, default_tdim)]

    if past:
        column = first(past)
        past_length_ = past[column].shape[tdims.get(column, default_tdim)]

        assert past_length is None or past_length == past_length_
        past_length = past_length_

    assert maybe.or_(past_length, future_length) is not None

    if full:
        column = first(full)
        full_length = full[column].shape[tdims.get(column, default_tdim)]

        assert full_length is None or full_length == full_length
        full_length = full_length

    if maybe.and_(past_length, future_length) is not None:
        if full_length is not None:
            assert full_length == past_length + future_length
    elif past_length is not None:
        assert full_length is not None

        future_length = full_length - past_length
    elif future_length is not None:
        assert full_length is not None

        past_length = full_length - future_length

    # create copies to not mutate user provided dicts
    past = dict(past)
    future = dict(future)

    for name, data in full.items():
        tdim = tdims.get(name, default_tdim)
        assert data.shape[tdim] == past_length + future_length

        past_data, future_data = np.split(data, [past_length], axis=tdim)
        past[name] = past_data
        future[name] = future_data

    sf = SplitFrame(
        _past=past,
        _future=future,
        index=index,
        static=static,
        tdims=tdims,
        past_length=past_length,
        future_length=future_length,
        default_tdim=default_tdim,
        metadata=metadata,
    )

    if sf.index is None and start is not None:
        if freq is not None:
            start = period(start, freq)

        return sf.with_index(start.periods(len(sf)))

    return sf
