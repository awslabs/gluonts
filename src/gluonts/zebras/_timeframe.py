from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from operator import methodcaller
from typing import Optional, List

import numpy as np
from toolz import first, valmap, dissoc

from gluonts import maybe

from ._period import Period, Periods
from ._util import pad_axis


@dataclass
class AxisView:
    data: np.ndarray
    axis: int

    def __len__(self):
        return self.data.shape[self.axis]

    def __getitem__(self, index):
        slices = [slice(None)] * self.data.ndim
        slices[self.axis] = index

        return self.data[tuple(slices)]


@dataclass
class TimeFrame:
    columns: dict
    index: Optional[Periods]
    length: int
    static: dict
    tdims: dict
    default_tdim: int = -1
    _pad: Optional[np.ndarray] = None

    def __post_init__(self):
        for column in self.columns:
            self.tdims.setdefault(column, self.default_tdim)

        dims = np.array(
            [
                self.columns[column].shape[self.tdims[column]]
                for column in self.columns
            ]
        )

        assert (dims == self.length).all(), (dims, self.length)
        assert self.index is None or len(self.index) == self.length
        assert self._pad is None or len(self._pad) == self.length

    def copy(self):
        return TimeFrame(
            columns=dict(self.columns),
            index=self.index,
            static=dict(self.static),
            tdims=dict(self.tdims),
            default_tdim=self.default_tdim,
            length=self.length,
            _pad=self._pad,
        )

    def head(self, count: int) -> Periods:
        return self[:count]

    def tail(self, count: int) -> Periods:
        if count is None:
            return self

        return self[-count:]

    def _cv(self, column):
        return AxisView(self.columns[column], self.tdims[column])

    def __getitem__(self, idx):
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
        )

    def split(self, idx, past_length=None, future_length=None, pad_value=0):
        if isinstance(idx, Period):
            idx = self.index.index_of(idx)

        past = self[slice(None, idx)].tail(past_length)
        future = self[slice(idx, None)].head(future_length)

        if past_length is not None:
            if len(past) < past_length:
                past = past.pad(pad_value, left=past_length - len(past))

        if future_length is not None and len(future) < future_length:
            future = future.pad(pad_value, right=future_length - len(future))

        return SplitFrame(past, future)

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

        pad = np.zeros(len(result))
        pad[:left] = 1
        pad[len(result) - right :] = 1

        if self._pad is not None:
            pad[left : len(result) - right :] = self._pad

        result._pad = pad

        if self.index is not None:
            result.index = self.index.prepend(left).extend(right)

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


@dataclass
class SplitFrame:
    past: TimeFrame
    future: TimeFrame

    def __len__(self):
        return len(self.past) + len(self.future)

    def set(self, name, data, tdim=-1):
        view = AxisView(data, tdim)
        assert len(view) == len(self)

        past = self.past.set(name, view[: len(self.past)], tdim)
        future = self.future.set(name, view[len(self.past) :], tdim)

        return SplitFrame(past, future)

    def set_past(self, name, data, tdim=-1, replace=False):
        return replace(self, past=self.past.set(name, data, tdim, replace))

    def set_future(self, name, data, tdim=-1, replace=False):
        return replace(self, future=self.future.set(name, data, tdim, replace))

    def shift(self, amount: int) -> SplitFrame:
        assert self.past.columns.keys() == self.future.columns.keys()

        ...

    def _view_of(self, column):
        is_past = column in self.past.columns
        is_future = column in self.future.columns

        if is_past and is_future:
            return self.set

        if is_past:
            return self.set_past

        if is_future:
            return self.set_future

        raise KeyError(f"Unknown column {column}.")

    def set_like(self, ref, name, data, tdim=-1, replace=False):
        return self._view_of(ref)(name, data, tdim, replace=False)

    @property
    def static(self):
        return self.past.static

    def as_dict(
        self, past="past", future="future", pad=None, static=True
    ) -> dict:
        result = {
            **{
                f"{past}_{name}": value
                for name, value in self.past.as_dict(pad, static=False).items()
            },
            **{
                f"{future}_{name}": value
                for name, value in self.future.as_dict(
                    pad, static=False
                ).items()
            },
        }
        if static:
            result.update(self.static)

        return result

    def c(self, name):
        past = self.past.columns.get(name)
        future = self.future.columns.get(name)

        if past is None:
            return future
        if future is None:
            return past

        return np.concatenate([past, future])

    @property
    def index(self):
        if self.past.index is None:
            return None
        assert self.future.index is not None

        return self.past.index.cat(self.future.index)

    def reindex(self, index):
        assert len(index) == len(self)
        self.past.reindex(index[: len(self.past)])
        self.future.reindex(index[len(self.past) :])


def split_frame(
    full=None,
    past=None,
    future=None,
    past_length=None,
    future_length=None,
    static=None,
    start=None,
    index=None,
    tdims=None,
    default_tdim=-1,
):
    full = maybe.unwrap_or_else(full, dict)
    past = maybe.unwrap_or_else(past, dict)
    future = maybe.unwrap_or_else(future, dict)
    static = maybe.unwrap_or_else(static, dict)
    tdims = maybe.unwrap_or_else(tdims, dict)

    if past_length is None:
        if past:
            column = first(past)
            past_length = past[column].shape[tdims.get(column, default_tdim)]
        else:
            raise ValueError("Cannot infer past length")

    if future_length is None:
        if future:
            column = first(future)
            future_length = future[column].shape[
                tdims.get(column, default_tdim)
            ]
        else:
            raise ValueError("Cannot infer future length")

    pf = SplitFrame(
        past=TimeFrame(
            columns=past,
            index=None,
            static=static,
            tdims=tdims,
            length=past_length,
            default_tdim=default_tdim,
        ),
        future=TimeFrame(
            columns=future,
            index=None,
            static=static,
            tdims=tdims,
            length=future_length,
            default_tdim=default_tdim,
        ),
    )

    if pf.index is None and start is not None:
        pf.reindex(start.periods(len(pf)))

    return pf


def time_frame(
    columns=None,
    index=None,
    start=None,
    static=None,
    tdims=None,
    length=None,
    default_tdim=-1,
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
    )
    if tf.index is None and start is not None:
        tf.reindex(start.periods(len(tf)))

    return tf
