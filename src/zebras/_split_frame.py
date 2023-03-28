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
from typing import Optional, List

import numpy as np
from toolz import first, keymap, valmap, dissoc, merge

from gluonts import maybe
from gluonts.itertools import pluck_attr, rows_to_columns

from ._base import Pad
from ._period import Periods, period
from ._repr import html_table
from ._util import _replace


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

    @staticmethod
    def _batch(split_frames: List[SplitFrame]) -> BatchSplitFrame:
        ref = split_frames[0]
        pluck = pluck_attr(split_frames)

        return BatchSplitFrame(
            _past=rows_to_columns(pluck("_past"), np.stack),
            _future=rows_to_columns(pluck("_future"), np.stack),
            index=pluck("index"),
            static=rows_to_columns(pluck("static"), np.stack),
            past_length=ref.past_length,
            future_length=ref.future_length,
            tdims=ref.tdims,
            metadata=pluck("metadata"),
            _pad=pluck("_pad"),
        )


@dataclasses.dataclass
class BatchSplitFrame:
    _past: dict
    _future: dict
    index: List[Optional[Periods]]
    static: dict
    past_length: int
    future_length: int
    tdims: dict
    metadata: List[Optional[dict]]
    _pad: List[Pad]

    @property
    def batch_size(self):
        return len(self.index)

    def __len__(self):
        return self.past_length + self.future_length

    @property
    def past(self):
        return BatchTimeFrame(
            columns=self._past,
            static=self.static,
            length=self.past_length,
            index=[
                maybe.map(index, itemgetter(slice(None, self.past_length)))
                for index in self.index
            ],
            tdims=self.tdims,
            metadata=self.metadata,
            _pad=self._pad,
        )

    @property
    def future(self):
        return BatchTimeFrame(
            columns=self._future,
            static=self.static,
            length=self.future_length,
            index=[
                maybe.map(index, itemgetter(slice(self.past_length, None)))
                for index in self.index
            ],
            tdims=self.tdims,
            metadata=self.metadata,
            _pad=self._pad,
        )

    def items(self):
        return BatchSplitFrameItems(self)

    def as_dict(self):
        return SplitFrame.as_dict(self)


@dataclasses.dataclass(repr=False)
class BatchSplitFrameItems:
    data: BatchSplitFrame

    def __len__(self):
        return self.data.batch_size

    def __getitem__(self, idx):
        tdims = self.data.tdims

        if isinstance(idx, int):
            cls = SplitFrame
            tdims = valmap(lambda tdim: tdim - 1 if tdim >= 0 else tdim, tdims)
        else:
            cls = BatchSplitFrame

        return cls(
            _past=valmap(itemgetter(idx), self.data._past),
            _future=valmap(itemgetter(idx), self.data._future),
            index=self.data.index[idx],
            static=valmap(itemgetter(idx), self.data.static),
            tdims=tdims,
            metadata=self.data.metadata[idx],
            _pad=self.data._pad[idx],
            past_length=self.data.past_length,
            future_length=self.data.future_length,
        )


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


# We defer these imports to avoid circular imports.
from ._time_series import TimeSeries  # noqa
from ._timeframe import BatchTimeFrame, TimeFrame  # noqa
