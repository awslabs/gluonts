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
from typing import Optional, List, NamedTuple

import numpy as np
from toolz import first, keymap, valmap, dissoc, merge, itemmap

from gluonts import maybe
from gluonts.itertools import pluck_attr, columns_to_rows, select

from ._period import Periods, Period, period
from ._repr import html_table
from ._util import AxisView, pad_axis


class Pad(NamedTuple):
    left: int = 0
    right: int = 0


@dataclasses.dataclass
class IndexView:
    tf: TimeFrame

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        assert isinstance(idx, slice)
        left, right, step = idx.indices(len(self.tf))
        assert step == 1

        pad_left = max(0, self.tf._pad.left - left)
        pad_right = max(0, self.tf._pad.right - right)

        index = maybe.map(self.tf.index, lambda index: index[idx])
        columns = {
            column: self.tf.vt(column)[idx] for column in self.tf.columns
        }

        return _replace(
            self.tf,
            index=index,
            columns=columns,
            length=right - left,
            _pad=Pad(pad_left, pad_right),
        )


@dataclasses.dataclass
class TimeView:
    tf: TimeFrame

    def __getitem__(self, idx):
        if isinstance(idx, Period):
            idx = slice(idx, idx + 1)

        assert isinstance(idx, slice)
        assert idx.step is None or idx.step == 1

        start = maybe.map(idx.start, self.tf.index_of)
        stop = maybe.map(idx.stop, self.tf.index_of)

        return IndexView(self.tf)[start:stop]


def _replace(obj, **kwargs):
    """Copy and replace dataclass instance.

    Compared to ``dataclasses.replace`` this first creates a copy where each
    field in the object is copied. Thus, each field of the returned object is
    different from the source object.
    """

    clone = object.__new__(obj.__class__)
    clone.__dict__ = valmap(copy.copy, obj.__dict__)

    return dataclasses.replace(clone, **kwargs)


@dataclasses.dataclass
class TimeFrame:
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
            assert len(self.vt(column)) == self.length, (
                f"Column {column!r} has incorrect length in time dimension. "
                f"Expected: {len(self)}, got {len(self.vt(column))}."
            )

        assert maybe.map_or(self.index, len, self.length) == self.length

    @property
    def start(self):
        return self.ix[0]

    @property
    def end(self):
        return self.ix[-1]

    def head(self, count: int) -> Periods:
        return self.ix[:count]

    def tail(self, count: int) -> Periods:
        if count is None:
            return self

        return self.ix[-count:]

    def vt(self, column):
        """View of column with respect to time."""

        return AxisView(self.columns[column], self.tdims[column])

    @property
    def tx(self):
        return TimeView(self)

    @property
    def ix(self):
        return IndexView(self)

    def __getitem__(self, col: str):
        return self.columns[col]

    def pad(self, value, left=0, right=0):
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
        if index is not None:
            index = self.index.prepend(left).extend(right)

        return _replace(
            self,
            columns=columns,
            index=index,
            length=length,
            _pad=Pad(pad_left, pad_right),
        )

    def index_of(self, period: Period):
        assert self.index is not None

        return self.index.index_of(period)

    def astype(self, type, columns=None):
        if columns is None:
            columns = self.columns

        return _replace(
            self,
            columns=valmap(
                lambda col: col.astype(type), select(columns, self.columns)
            ),
        )

    def __len__(self):
        return self.length

    def __repr__(self):
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

    def with_index(self, index):
        return _replace(self, index=index)

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
            index=self.index,
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

        length = max(len(first(past.values())), len(first(future.values())))

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


def time_frame(
    columns=None,
    *,
    index=None,
    start=None,
    freq=None,
    static=None,
    tdims=None,
    length=None,
    default_tdim=-1,
    metadata=None,
):
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
            raise ValueError("Cannot infer length.")

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
