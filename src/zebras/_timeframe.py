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
from typing import Optional, List, Union, Collection, Dict, Any

import numpy as np
from toolz import first, valmap, dissoc, merge, itemmap

from gluonts import maybe
from gluonts.itertools import (
    pluck_attr,
    columns_to_rows,
    rows_to_columns,
    select,
)

from ._base import Pad, TimeBase
from ._period import Periods, Period, period
from ._repr import html_table
from ._util import AxisView, pad_axis, _replace


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

    def _batch(self, xs):
        # TODO: Check
        ref = xs[0]
        pluck = pluck_attr(xs)

        tdims = valmap(
            lambda tdim: tdim + 1 if tdim >= 0 else tdim,
            ref.tdims,
        )

        return BatchTimeFrame(
            columns=rows_to_columns(pluck("columns"), np.stack),
            index=pluck("index"),
            static=rows_to_columns(pluck("static"), np.stack),
            length=ref.length,
            tdims=tdims,
            metadata=pluck("metadata"),
            _pad=pluck("_pad"),
        )


@dataclasses.dataclass
class BatchTimeFrame:
    columns: dict
    index: Collection[Optional[Periods]]
    static: dict
    length: int
    tdims: dict
    metadata: Collection[Optional[dict]]
    _pad: Collection[Pad]

    @property
    def batch_size(self):
        return len(self.index)

    def like(self, columns=None, static=None, tdims=None):
        columns = maybe.unwrap_or(columns, {})
        static = maybe.unwrap_or(static, {})

        tdims = maybe.unwrap_or(tdims, {})
        for name in columns:
            tdims.setdefault(name, -1)

        return _replace(
            self, columns=columns, index=self.index, static=static, tdims=tdims
        )

    def items(self):
        return BatchTimeFrameItems(self)

    def __getitem__(self, name):
        return BatchTimeSeries(
            self.columns[name],
            index=self.index,
            tdim=self.tdims[name],
            metadata=self.metadata,
            name=name,
            _pad=self._pad,
        )

    def as_dict(self, prefix=None, static=True):
        return TimeFrame.as_dict(self, prefix, static)


@dataclasses.dataclass(repr=False)
class BatchTimeFrameItems:
    data: BatchTimeFrame

    def __len__(self):
        return self.data.batch_size

    def __getitem__(self, idx):
        tdims = self.data.tdims

        if isinstance(idx, int):
            cls = TimeFrame
            tdims = valmap(lambda tdim: tdim - 1 if tdim >= 0 else tdim, tdims)
        else:
            cls = BatchTimeFrame

        return cls(
            columns=valmap(itemgetter(idx), self.data.columns),
            index=self.data.index[idx],
            static=valmap(itemgetter(idx), self.data.static),
            tdims=tdims,
            metadata=self.data.metadata[idx],
            _pad=self.data._pad[idx],
            length=self.data.length,
        )


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


# We defer importing `TimeSeries` to avoid circular imports.
from ._time_series import BatchTimeSeries, TimeSeries  # noqa
from ._split_frame import SplitFrame  # noqa
