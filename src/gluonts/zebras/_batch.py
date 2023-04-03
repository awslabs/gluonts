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


import dataclasses
from typing import Optional, Collection

import numpy as np
from toolz import compose_left, curry

from gluonts import maybe
from gluonts.itertools import rows_to_columns, columns_to_rows, pluck_attr

from ._period import Periods
from ._timeframe import TimeFrame, SplitFrame, Pad, _replace, BatchTimeSeries


@dataclasses.dataclass
class BatchTimeFrame:
    columns: dict
    index: Collection[Optional[Periods]]
    static: dict
    length: int
    tdims: dict
    metadata: Collection[Optional[dict]]
    _pad: Collection[Pad]

    def like(self, columns=None, static=None, tdims=None):
        columns = maybe.unwrap_or(columns, {})
        static = maybe.unwrap_or(static, {})

        tdims = maybe.unwrap_or(tdims, {})
        for name in columns:
            tdims.setdefault(name, -1)

        return _replace(
            self, columns=columns, index=self.index, static=static, tdims=tdims
        )

    def __iter__(self):
        if self.static:
            statics = columns_to_rows(self.static)
        else:
            statics = [{}] * self.length

        for (
            index,
            columns,
            static,
            metadata,
            pad,
        ) in zip(
            self.index,
            columns_to_rows(self.columns),
            statics,
            self.metadata,
            self._pad,
        ):
            yield TimeFrame(
                columns=columns,
                index=index,
                static=static,
                length=self.length,
                tdims=self.tdims,
                metadata=metadata,
                _pad=pad,
            )

    def __getitem__(self, name):
        return BatchTimeSeries(
            self.columns[name],
            index=self.index,
            tdim=self.tdims[name],
            metadata=self.metadata,
            name=name,
            _pad=self._pad,
        )


@dataclasses.dataclass
class BatchSplitFrame:
    _past: dict
    _future: dict
    index: Optional[Periods]
    static: dict
    past_length: int
    future_length: int
    tdims: dict
    metadata: Collection[Optional[dict]]
    _pad: Collection[Pad]

    def __post_init__(self):
        for index in self.index:
            assert (
                index is None
                or len(index) == self.past_length + self.future_length
            ), f"Index length: {len(index)}, expected: {len(self)}"

    def __len__(self):
        return self.past_length + self.future_length

    @property
    def past(self):
        return BatchTimeFrame(
            columns=self._past,
        )

    @property
    def future(self):
        return BatchTimeFrame(
            columns=self._future,
            static=self.static,
            length=self.future_length,
            index=[index[self.past_length :] for index in self.index],
            tdims=self.tdims,
            metadata=self.metadata,
            _pad=self._pad,
        )

    def as_dict(self):
        return SplitFrame.as_dict(self)


def batch_splitframe(frames, type):
    ref = frames[0]
    get = pluck_attr(frames)

    return BatchSplitFrame(
        _past=rows_to_columns(get("_past"), compose_left(np.stack, type)),
        _future=rows_to_columns(get("_future"), compose_left(np.stack, type)),
        index=get("index"),
        static=rows_to_columns(get("static"), compose_left(np.stack, type)),
        past_length=ref.past_length,
        future_length=ref.future_length,
        tdims=ref.tdims,
        metadata=get("metadata"),
        _pad=get("_pad"),
    )


def batch_timeframe(frames, type):
    ref = frames[0]
    get = pluck_attr(frames)

    return BatchTimeFrame(
        columns=rows_to_columns(get("columns"), compose_left(np.stack, type)),
        _future=rows_to_columns(get("_future"), compose_left(np.stack, type)),
        index=get("index"),
        static=rows_to_columns(get("static"), compose_left(np.stack, type)),
        length=ref.past_length,
        tdims=ref.tdims,
        metadata=get("metadata"),
        _pad=get("_pad"),
    )


@curry
def batch(frames, type=np.array):
    assert frames
    if isinstance(frames[0], TimeFrame):
        return batch_timeframe(frames, type)

    return batch_splitframe(frames, type)
