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
import itertools
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

    def __post_init__(self):
        for column in itertools.chain(self._past, self._future):
            self.tdims.setdefault(column, self.default_tdim)

        # this triggers checks for past_length and future_length
        _, _ = self.past, self.future

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
                self.index, lambda index: index[self.past_length :]
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
            # Calculate new start. If current past_length is larger than the
            # the new one, we shift it to the right, if it's smaller, we need
            # to go further into the past (shift to the left)
            start = index[0] + (self.past_length - past_length)
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
            _past=rows_to_columns(pluck("_past"), np.stack),  # type: ignore
            _future=rows_to_columns(pluck("_future"), np.stack),  # type: ignore
            index=pluck("index"),
            static=rows_to_columns(pluck("static"), np.stack),  # type: ignore
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
    static=None,
    index=None,
    start=None,
    freq=None,
    tdims=None,
    metadata=None,
    default_tdim=-1,
):
    """Create a ``zebras.SplitFrame`` where columns can either be `past`,
    `future` or `full`, which spans both past and future.

    ``past_length`` and ``future_length`` is derived from the input data if
    possible or default to ``0`` in case no respective data is available. It is
    possible to set these values explicitly for enforcing consistency or to
    provide a length even though no time series spans that range.

    Parameters
    ----------
    full, optional
        Time series columns that span past and future.
    past, optional
        Time series columns that are past only.
    future, optional
        Time series columns that are future only.
    past_length, optional
        Set length of the past section, derived from data if not provided.
    future_length, optional
        Set length of the future section, derived from data if not provided.
    static, optional
        Values that are independent of time.
    index, optional
        A ``zebras.Periods`` object representing timestamps.
        Must have the same length as full range.
    start, optional
        The start time represented by a string (e.g., "2023-01-01"),
        or a ``zebras.Period`` object. An index will be constructed using
        this start time and the specified frequency
    freq, optional
        The frequency to use for constructing the index.
    tdims, optional
        A dictionary specifying the time dimension for each column, this
        applies to past, future and full.
    default_tdim, optional
        The default time dimension, by default -1
    metadata, optional
        A dictionary of metadata associated with the TimeFrame, by default None
    Returns
    -------
        A ``zebras.SplitFrame`` object.
    """
    full = valmap(np.array, maybe.unwrap_or_else(full, dict))
    past = valmap(np.array, maybe.unwrap_or_else(past, dict))
    future = valmap(np.array, maybe.unwrap_or_else(future, dict))
    static = valmap(np.array, maybe.unwrap_or_else(static, dict))
    tdims = maybe.unwrap_or_else(tdims, dict)

    # Resolve `past_length` and `future_length` if not directly set from
    # provided data. If no data is passed for either field, the value is still
    # `None` after this.
    if past_length is None and past:
        column = first(past)
        past_length = past[column].shape[tdims.get(column, default_tdim)]

    if future_length is None and future:
        column = first(future)
        future_length = future[column].shape[tdims.get(column, default_tdim)]

    if full:
        column = first(full)
        full_length = full[column].shape[tdims.get(column, default_tdim)]

        if maybe.or_(past_length, future_length) is None:
            raise ValueError(
                "Cannot determine past and future length if only "
                "`full` is provided."
            )
    # No data and no lengths are passed
    elif maybe.or_(past_length, future_length) is None:
        past_length = 0
        future_length = 0
        full_length = 0
    else:
        # If past_length and/or future_length is provided, but no `full` data
        # is given, then we first resolve past and future length and then just
        # calculate full_length later.
        full_length = None

    if past_length is None:
        past_length = maybe.map_or(
            full_length,
            lambda fl: fl - future_length,
            0,
        )
    elif future_length is None:
        future_length = maybe.map_or(
            full_length,
            lambda fl: fl - past_length,
            0,
        )
    full_length = past_length + future_length

    # create copies to not mutate user provided dicts
    past = dict(past)
    future = dict(future)

    for name, data in full.items():
        tdim = tdims.get(name, default_tdim)
        assert data.shape[tdim] == full_length

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
from ._time_frame import BatchTimeFrame, TimeFrame  # noqa
