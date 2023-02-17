from __future__ import annotations

import copy
import functools
import dataclasses
from operator import methodcaller
from typing import Optional, List

import numpy as np
from toolz import first, valmap, dissoc

from gluonts import maybe
from gluonts.itertools import rows_to_columns, pluck_attr

from ._period import Period, Periods
from ._util import pad_axis


def replace(obj, **kwargs):
    """Copy and replace dataclass instance.

    Compared to ``dataclasses.replace`` this first creates a copy where each
    field in the object is copied. Thus, each field of the returned object is
    different from the source object.
    """

    clone = object.__new__(obj.__class__)
    clone.__dict__ = valmap(copy.copy, obj.__dict__)

    return dataclasses.replace(clone, **kwargs)


@dataclasses.dataclass
class AxisView:
    data: np.ndarray
    axis: int

    def __len__(self):
        return self.data.shape[self.axis]

    def __getitem__(self, index):
        slices = [slice(None)] * self.data.ndim
        slices[self.axis] = index

        return self.data[tuple(slices)]


@dataclasses.dataclass
class TimeFrame:
    columns: dict
    index: Optional[Periods]
    length: int
    static: dict
    tdims: dict
    metadata: Optional[dict] = None
    default_tdim: int = -1
    _pad: Tuple[int, int] = (0, 0)

    def __post_init__(self):
        for column in self.columns:
            self.tdims.setdefault(column, self.default_tdim)

        dims = np.array(
            [
                self.columns[column].shape[self.tdims[column]]
                for column in self.columns
            ]
        )

    @property
    def start(self):
        return self[0]

    @property
    def end(self):
        return self[-1]

    def head(self, count: int) -> Periods:
        return self[:count]

    def tail(self, count: int) -> Periods:
        if count is None:
            return self

        return self[-count:]

    def _cv(self, column):
        return AxisView(self.columns[column], self.tdims[column])

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.c(idx)

        if isinstance(idx, int):
            return {column: self._cv(column)[idx] for column in self.columns}

        start, stop, step = idx.indices(len(self))
        assert step == 1

        columns = {column: self._cv(column)[idx] for column in self.columns}

        return TimeFrame(
            columns=columns,
            index=maybe.map(self.index, methodcaller("__getitem__", idx)),
            static=self.static,
            tdims=self.tdims,
            default_tdim=self.default_tdim,
            length=stop - start,
            metadata=self.metadata,
        )

    def pad(self, value, left=0, right=0):
        assert left >= 0 and right >= 0
        result = self.copy()

        result.columns = {
            column: pad_axis(
                self.columns[column],
                axis=self.tdims[column],
                left=left,
                right=right,
                value=value,
            )
            for column in self.columns
        }
        result.length += left + right

        result._pad = (left + self._pad[0], right + self._pad[1])

        if self.index is not None:
            result.index = self.index.prepend(left).extend(right)

        return result
        assert (dims == self.length).all(), (dims, self.length)
        assert self.index is None or len(self.index) == self.length

    def like(self, columns=None, static=None, tdims=None):
        columns = maybe.unwrap_or(columns, {})
        static = maybe.unwrap_or(static, {})
        tdims = maybe.unwrap_or(tdims, {})

        return TimeFrame(
            columns=columns,
            index=self.index,
            static=static,
            tdims=tdims,
            default_tdim=self.default_tdim,
            length=self.length,
            metadata=self.metadata,
            _pad=self._pad,
        )

    def astype(self, type):
        result = self.copy()
        result.columns = valmap(type, self.columns)
        result.static = valmap(type, self.static)
        return result

    def __len__(self):
        return self.length

    def __repr__(self):
        columns = ", ".join(self.columns)
        return f"TimeFrame<size={len(self)}, columns=[{columns}]>"

    def _repr_html_(self):
        def tag(name, data):
            return f"<{name}>{data}</{name}>"

        th = functools.partial(tag, "th")
        td = functools.partial(tag, "td")
        tr = functools.partial(tag, "tr")

        if self.index is not None:
            index_th = [""]
        else:
            index_th = []

        thead = " ".join(list(map(th, index_th + list(self.columns))))

        rows = []
        for idx in range(len(self)):
            if self.index is not None:
                index_td = [td(self.index[idx].data)]
            else:
                index_td = []

            rows.append(
                tr(" ".join(index_td + list(map(td, self[idx].values()))))
            )

        return f"""
        <table>
            <thead>
                {thead}
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>

        {len(self)} rows × {len(self.columns)} columns
        """

    def set(self, name, value, tdim=None, replace=False):
        assert replace or name not in self.columns

        tdim = maybe.unwrap_or(tdim, self.default_tdim)
        view = AxisView(value, tdim)
        assert len(view) == len(self)

        result = self.copy()

        result.columns[name] = value
        result.tdims[name] = tdim

        return result

    def remove(self, column):
        result = self.copy()
        result.columns.pop(column)
        result.tdims.pop(column, None)

        return result

    def stack(
        self,
        columns: List[str],
        into: str,
        drop: bool = True,
        replace: bool = False,
    ) -> TimeFrame:
        assert replace or into not in self.columns

        assert len(set([self.tdims[column] for column in columns])) == 1

        result = self.copy()

        if drop:
            result.columns = dissoc(result.columns, *columns)
            result.tdims = dissoc(result.tdims, *columns)

        return result.set(
            into,
            np.vstack([self.columns[column] for column in columns]),
            replace=True,
        )

    def set_like(self, ref, name, data, tdim=-1, replace=False):
        return self.set(name, data, tdim, replace)

    def c(self, name):
        return self.columns[name]

    def reindex(self, index):
        assert len(index) == len(self)
        self.index = index

    def as_dict(self, prefix=None, pad=None, static=True):
        result = dict(self.columns)

        if prefix is not None:
            result = {prefix + key: value for key, value in result.items()}

        if static:
            result.update(self.static)

        if pad is not None:
            result[pad] = self._pad

        return result

    def cat(self, other):
        assert self.columns.keys() == other.columns.keys()
        assert self.tdims == other.tdims
        assert maybe.xor(self.index, other.index) is None

        copy = self.copy()
        copy.length += len(other)
        copy.columns = {
            column: np.concatenate(
                [self.columns[column], other.columns[column]],
                axis=self.tdims[column],
            )
            for column in self.columns
        }
        if self.index is not None:
            copy.index = np.concatenate([self.index, other.index])

        return copy


def time_frame(
    columns=None,
    index=None,
    start=None,
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
        tf.reindex(start.periods(len(tf)))

    return tf
