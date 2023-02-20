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
from toolz import first, valmap, dissoc, merge

from gluonts import maybe
from gluonts.itertools import pluck_attr

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
            assert len(self.vt(column)) == self.length

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
            _pad=(pad_left, pad_right),
        )

    def index_of(self, period: Period):
        assert self.index is not None

        return self.index.index_of(period)

    def astype(self, type):
        return _replace(
            self,
            columns=valmap(type, self.columns),
            static=valmap(type, self.static),
        )

    def __len__(self):
        return self.length

    def __repr__(self):
        columns = ", ".join(self.columns)
        return f"TimeFrame<size={len(self)}, columns=[{columns}]>"

    def _repr_html_(self):
        columns = {}

        if self.index is not None:
            columns[""] = pluck_attr(self.index, "data")

        columns.update(self.columns)

        table = html_table(columns)
        return table + f"{len(self)} rows × {len(self.columns)} columns"

    def set(self, name, value, tdim=None):
        tdim = maybe.unwrap_or(tdim, self.default_tdim)

        return _replace(
            columns=merge(self.columns, {name: value}),
            tdims=merge(self.tdims, {name: tdim}),
        )

    def remove(self, column):
        return _replace(
            self,
            columns=dissoc(self.columns, column),
            tdims=dissoc(self.tdims, column),
        )

    def stack(
        self,
        columns: List[str],
        into: str,
        drop: bool = True,
    ) -> TimeFrame:
        # Ensure all tdims are the same.
        # TODO: Can we make that work for different tdims? There might be a
        # problem with what the resulting dimensions are.
        assert len(set([self.tdims[column] for column in columns])) == 1

        if drop:
            columns = dissoc(self.columns, *columns)
            tdims = dissoc(self.tdims, *columns)
        else:
            columns = self.columns
            tdims = self.tdims

        columns[into] = np.vstack([self.columns[column] for column in columns])

        return _replace(self, columns=columns, tdims=tdims)

    def with_index(self, index):
        return _replace(self, index=index)

    def as_dict(self, prefix=None, pad=None, static=True):
        result = dict(self.columns)

        if prefix is not None:
            result = {prefix + key: value for key, value in result.items()}

        if static:
            result.update(self.static)

        if pad is not None:
            result[pad] = self._pad

        return result


def time_frame(
    columns=None,
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
