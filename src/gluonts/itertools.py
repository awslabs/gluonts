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

import itertools
import math
import pickle
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Sequence,
    Tuple,
    NamedTuple,
)
from typing_extensions import Protocol, runtime_checkable
from numpy.random import RandomState
from toolz import curry

import numpy as np


@runtime_checkable
class SizedIterable(Protocol):
    def __len__(self): ...

    def __iter__(self): ...


T = TypeVar("T")

# key / value
K = TypeVar("K")
V = TypeVar("V")


def maybe_len(obj) -> Optional[int]:
    try:
        return len(obj)
    except (NotImplementedError, AttributeError):
        return None


def prod(xs):
    """
    Compute the product of the elements of an iterable object.
    """
    p = 1
    for x in xs:
        p *= x
    return p


@dataclass
class Cyclic:
    """
    Like `itertools.cycle`, but does not store the data.
    """

    iterable: SizedIterable

    def __iter__(self):
        at_least_one = False
        while True:
            for el in self.iterable:
                at_least_one = True
                yield el
            if not at_least_one:
                break

    def stream(self):
        """
        Return a continuous stream of self that has no fixed start.

        When re-iterating ``Cyclic`` it will yield elements from the start of
        the passed ``iterable``. However, this is not always desired; e.g. in
        training we want to treat training data as an infinite stream of
        values and not start at the beginning of the dataset for each epoch.

        >>> from toolz import take
        >>> c = Cyclic([1, 2, 3, 4])
        >>> assert list(take(5, c)) == [1, 2, 3, 4, 1]
        >>> assert list(take(5, c)) == [1, 2, 3, 4, 1]

        >>> s = Cyclic([1, 2, 3, 4]).stream()
        >>> assert list(take(5, s)) == [1, 2, 3, 4, 1]
        >>> assert list(take(5, s)) == [2, 3, 4, 1, 2]
        """
        return iter(self)

    def __len__(self) -> int:
        return len(self.iterable)


def batcher(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """
    Groups elements from `iterable` into batches of size `batch_size`.

    >>> list(batcher("ABCDEFG", 3))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]

    Unlike the grouper proposed in the documentation of itertools, `batcher`
    doesn't fill up missing values.
    """
    it: Iterator[T] = iter(iterable)

    def get_batch():
        return list(itertools.islice(it, batch_size))

    # has an empty list so that we have a 2D array for sure
    return iter(get_batch, [])


@dataclass
class Chain:
    """
    Chain multiple iterables into a single one.

    This is a thin wrapper around ``itertools.chain``.
    """

    iterables: Collection[SizedIterable]

    def __iter__(self):
        yield from itertools.chain.from_iterable(self.iterables)

    def __len__(self) -> int:
        return sum(map(len, self.iterables))


class _SubIndex(NamedTuple):
    """
    A nested index with two layers.
    """

    item: int
    local: int


@dataclass
class Fuse:
    """
    Fuse collections together to act as single collections.

    >>> a = [0, 1, 2]
    >>> b = [3, 4, 5]
    >>> fused = Fuse([a, b])
    >>> assert len(a) + len(b) == len(fused)
    >>> list(fused[2:5])
    [2, 3, 4]
    >>> list(fused[3:])
    [3, 4, 5]

    This is similar to ``Chain``, but also allows to select directly into the
    data. While ``Chain`` creates a single ``Iterable`` out of a set of
    ``Iterable``s, ``Fuse`` creates a single ``Collection`` out of a set of
    ``Collection``s.
    """

    collections: List[Sequence]
    _lengths: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self._lengths:
            self._lengths = list(map(len, self.collections))

        self._length = sum(self._lengths)
        self._offsets = np.cumsum(self._lengths)

    def __len__(self):
        return self._length

    def _get_range(self, start: _SubIndex, stop: _SubIndex) -> "Fuse":
        first = self.collections[start.item]

        if start.item == stop.item:
            return Fuse([first[start.local : stop.local]])

        items = []

        first = first[start.local :]
        if len(first) > 0:
            items.append(first)

        for item_index in range(start.item + 1, stop.item):
            items.append(self.collections[item_index])

        items.append(self.collections[stop.item][: stop.local])
        return Fuse(items)

    def _location_for(self, idx, side="right") -> _SubIndex:
        """
        Map global index to pair of index to collection and index within that
        collection.

        >>> fuse = Fuse([[0, 0], [1, 1]])
        >>> fuse._location_for(0)
        _SubIndex(item=0, local=0)
        >>> fuse._location_for(1)
        _SubIndex(item=0, local=1)
        >>> fuse._location_for(2)
        _SubIndex(item=1, local=0)
        >>> fuse._location_for(3)
        _SubIndex(item=1, local=1)
        """
        if idx == 0 or not self:
            return _SubIndex(0, 0)

        # When the index is out of bounds, we fall back to the last element
        if idx >= len(self):
            return _SubIndex(
                len(self.collections) - 1,
                len(self.collections[-1]),
            )

        part_no = np.searchsorted(self._offsets, idx, side)

        if part_no == 0:
            local_idx = idx
        else:
            local_idx = idx - self._offsets[part_no - 1]

        return _SubIndex(int(part_no), int(local_idx))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            assert step is None or step == 1

            start_index = self._location_for(start)
            stop_index = self._location_for(stop)
            return self._get_range(start_index, stop_index)

        subindex = self._location_for(idx)
        return self.collections[subindex.item][subindex.local]

    def __iter__(self):
        yield from itertools.chain.from_iterable(self.collections)

    def __repr__(self):
        return f"Fuse<size={len(self)}>"


def split(xs: Sequence, indices: List[int]) -> List[Sequence]:
    """
    Split ``xs`` into subsets given ``indices``.

    >>> split("abcdef", [1, 3])
    ['a', 'bc', 'def']

    This is similar to ``numpy.split`` when passing a list of indices, but this
    version does not convert the underlying data into arrays.
    """

    return [
        xs[start:stop]
        for start, stop in zip(
            itertools.chain([None], indices),
            itertools.chain(indices, [None]),
        )
    ]


def split_into(xs: Sequence, n: int) -> Sequence:
    """
    Split ``xs`` into ``n`` parts of similar size.

    >>> split_into("abcd", 2)
    ['ab', 'cd']
    >>> split_into("abcd", 3)
    ['ab', 'c', 'd']
    """

    bucket_size, remainder = divmod(len(xs), n)

    # We need one fewer than `n`, since these become split positions.
    relative_splits = np.full(n - 1, bucket_size)
    # e.g. 10 by 3 -> 4, 3, 3
    relative_splits[:remainder] += 1

    return split(xs, np.cumsum(relative_splits))  # type: ignore[arg-type]


@dataclass
class Cached:
    """
    An iterable wrapper, which caches values in a list while iterated.

    The primary use-case for this is to avoid re-computing the elements of the
    sequence, in case the inner iterable does it on demand.

    This should be used to wrap deterministic iterables, i.e. iterables where
    the data generation process is not random, and that yield the same
    elements when iterated multiple times.
    """

    iterable: SizedIterable
    provider: Iterable = field(init=False)
    consumed: list = field(default_factory=list, init=False)

    def __post_init__(self):
        # ensure we only iterate once over the iterable
        self.provider = iter(self.iterable)

    def __iter__(self):
        # Yield already provided values first
        yield from self.consumed

        # Now yield remaining elements.
        for element in self.provider:
            self.consumed.append(element)
            yield element

    def __len__(self) -> int:
        return len(self.iterable)


@dataclass
class PickleCached:
    """
    A caching wrapper for ``iterable`` using ``pickle`` to store cached values
    on disk.

    See :class:`Cached` for more information.
    """

    iterable: SizedIterable
    cached: bool = False
    _path: Path = field(
        default_factory=lambda: Path(
            tempfile.NamedTemporaryFile(delete=False).name
        )
    )

    def __iter__(self):
        if not self.cached:
            with open(self._path, "wb") as tmpfile:
                for batch in batcher(self.iterable, 16):
                    pickle.dump(batch, tmpfile)
                    yield from batch
            self.cached = True
        else:
            with open(self._path, "rb") as tmpfile:
                while True:
                    try:
                        yield from pickle.load(tmpfile)
                    except EOFError:
                        return

    def __del__(self):
        self._path.unlink()


@dataclass
class PseudoShuffled:
    """
    Yield items from a given iterable in a pseudo-shuffled order.
    """

    iterable: SizedIterable
    shuffle_buffer_length: int

    def __iter__(self):
        shuffle_buffer = []

        for element in self.iterable:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                yield shuffle_buffer.pop(random.randrange(len(shuffle_buffer)))

        while shuffle_buffer:
            yield shuffle_buffer.pop(random.randrange(len(shuffle_buffer)))

    def __len__(self) -> int:
        return len(self.iterable)


# can't make this a dataclass because of pytorch-lightning assumptions
class IterableSlice:
    """
    An iterable version of `itertools.islice`, i.e. one that can be iterated
    over multiple times:

        >>> isl = IterableSlice(iter([1, 2, 3, 4, 5]), 3)
        >>> list(isl)
        [1, 2, 3]
        >>> list(isl)
        [4, 5]
        >>> list(isl)
        []

    This needs to be a class to support re-entry iteration.
    """

    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length

    def __iter__(self):
        yield from itertools.islice(self.iterable, self.length)


class SizedIterableSlice(IterableSlice):
    """
    Same as ``IterableSlice`` but also supports `len()`:

        >>> isl = SizedIterableSlice([1, 2, 3, 4, 5], 3)
        >>> len(isl)
        3
    """

    def __len__(self):
        # NOTE: This works correctly only when self.iterable supports `len()`.
        total_len = len(self.iterable)
        return min(total_len, self.length)


@dataclass
class Map:
    fn: Callable
    iterable: SizedIterable

    def __iter__(self):
        return map(self.fn, self.iterable)

    def __len__(self):
        return len(self.iterable)


@dataclass
class StarMap:
    fn: Callable
    iterable: SizedIterable

    def __iter__(self):
        return itertools.starmap(self.fn, self.iterable)

    def __len__(self):
        return len(self.iterable)


@dataclass
class Filter:
    fn: Callable
    iterable: SizedIterable

    def __iter__(self):
        return filter(self.fn, self.iterable)


@dataclass
class RandomYield:
    """
    Given a list of Iterables `iterables`, generate samples from them.

    When `probabilities` is given, sample iteratbles according to it.
    When `probabilities` is not given, sample iterables uniformly.

    When one iterable exhausts, the sampling probabilities for it will be set to 0.

        >>> from toolz import take
        >>> a = [1, 2, 3]
        >>> b = [4, 5, 6]
        >>> it = iter(RandomYield([a, b], probabilities=[1, 0]))
        >>> list(take(5, it))
        [1, 2, 3]


        >>> a = [1, 2, 3]
        >>> b = [4, 5, 6]
        >>> it = iter(RandomYield([Cyclic(a), b], probabilities=[1, 0]))
        >>> list(take(5, it))
        [1, 2, 3, 1, 2]
    """

    iterables: List[Iterable]
    probabilities: Optional[List[float]] = None
    random_state: RandomState = field(default_factory=RandomState)

    def __post_init__(self):
        if not self.probabilities:
            self.probabilities = [1.0 / len(self.iterables)] * len(
                self.iterables
            )

    def __iter__(self):
        iterators = [iter(it) for it in self.iterables]
        probs = list(self.probabilities)

        while True:
            idx = self.random_state.choice(range(len(iterators)), p=probs)
            try:
                yield next(iterators[idx])
            except StopIteration:
                probs[idx] = 0
                if sum(probs) == 0:
                    return
                probs = [prob / sum(probs) for prob in probs]


def rows_to_columns(
    rows: Sequence[Dict[K, V]],
    wrap: Callable[[Sequence[V]], Sequence[V]] = lambda x: x,
) -> Dict[K, Sequence[V]]:
    """
    Transpose rows of dicts, to one dict containing columns.

    >>> rows_to_columns([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    {'a': [1, 3], 'b': [2, 4]}

    This can also be understood as stacking the values of each dict onto each
    other.
    """

    if not rows:
        return {}

    column_names = rows[0].keys()

    return {
        column_name: wrap([row[column_name] for row in rows])
        for column_name in column_names
    }


def columns_to_rows(columns: Dict[K, Sequence[V]]) -> List[Dict[K, V]]:
    """
    Transpose column-orientation to row-orientation.

    >>> columns_to_rows({'a': [1, 3], 'b': [2, 4]})
    [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    """

    if not columns:
        return []

    column_names = columns.keys()

    return [
        dict(zip(column_names, values)) for values in zip(*columns.values())
    ]


def roundrobin(*iterables):
    """
    `roundrobin('ABC', 'D', 'EF') --> A D E B F C`

    Taken from: https://docs.python.org/3/library/itertools.html#recipes.
    """

    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def partition(
    it: Iterable[T], fn: Callable[[T], bool]
) -> Tuple[List[T], List[T]]:
    """
    Partition `it` into two lists given predicate `fn`.

    This is similar to the recipe defined in Python's `itertools` docs, however
    this method consumes the iterator directly  and returns lists instead of
    iterators.
    """

    left, right = [], []

    for val in it:
        if fn(val):
            left.append(val)
        else:
            right.append(val)

    return left, right


def select(keys, source: dict, ignore_missing: bool = False) -> dict:
    """
    Select subset of `source` dictionaries.

    >>> d = {"a": 1, "b": 2, "c": 3}
    >>> select(["a", "b"], d)
    {'a': 1, 'b': 2}
    """

    result = {}

    for key in keys:
        try:
            result[key] = source[key]
        except KeyError:
            if not ignore_missing:
                raise

    return result


def trim_nans(xs, trim="fb"):
    """
    Trim the leading and/or trailing `NaNs` from a 1-D array or sequence.

    Like ``np.trim_zeros`` but for `NaNs`.
    """

    trim = trim.lower()

    start = None
    end = None

    if "f" in trim:
        for start, val in enumerate(xs):
            if not math.isnan(val):
                break

    if "b" in trim:
        for end in range(len(xs), -1, -1):
            if not math.isnan(xs[end - 1]):
                break

    return xs[start:end]


def inverse(dct: Dict[K, V]) -> Dict[V, K]:
    """
    Inverse a dictionary; keys become values and values become keys.
    """
    return {value: key for key, value in dct.items()}


_no_default = object()


@curry
def pluck_attr(seq, name, default=_no_default):
    """
    Get attribute ``name`` from elements in ``seq``.
    """

    if default is _no_default:
        return [getattr(el, name) for el in seq]

    return [getattr(el, name, default) for el in seq]


def power_set(iterable):
    """
    Generate all possible subsets of the given iterable, as tuples.

    >>> list(power_set(["a", "b"]))
    [(), ('a',), ('b',), ('a', 'b')]

    Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    return itertools.chain.from_iterable(
        itertools.combinations(iterable, r) for r in range(len(iterable) + 1)
    )


def join_items(left, right, how="outer", default=None):
    """
    Iterate over joined dictionary items.

    Yields triples of `key`, `left_value`, `right_value`.

    Similar to SQL join statements the join behaviour is controlled by ``how``:

    * ``outer`` (default): use keys from left and right
    * ``inner``: only use keys which appear in both left and right
    * ``strict``: like ``inner``, but throws error if keys mismatch
    * ``left``: use only keys from ``left``
    * ``right``: use only keys from ``right``

    If a key is not present in either input, ``default`` is chosen instead.
    """

    if how == "outer":
        keys = {**left, **right}
    elif how == "strict":
        assert left.keys() == right.keys()
        keys = left.keys()
    elif how == "inner":
        keys = left.keys() & right.keys()
    elif how == "left":
        keys = left.keys()
    elif how == "right":
        keys = right.keys()
    else:
        raise ValueError(f"Unknown how={how}.")

    for key in keys:
        yield key, left.get(key, default), right.get(key, default)


def replace(values: Sequence[T], idx: int, value: T) -> Sequence[T]:
    """
    Replace value at index ``idx`` with ``value``.

    Like ``setitem``, but for tuples.

    >>> replace((1, 2, 3, 4), -1, 99)
    (1, 2, 3, 99)
    """
    xs = list(values)
    xs[idx] = value

    return type(values)(xs)  # type: ignore


def chop(
    at: int,
    take: int,
) -> slice:
    """
    Create slice using an index ``at`` and amount ``take``.

    >>> x = [0, 1, 2, 3, 4]
    >>> x[chop(at=1, take=2)]
    [1, 2]
    >>>
    >>> x[chop(at=-2, take=2)]
    [3, 4]
    >>> x[chop(at=3, take=-2)]
    [1, 2]
    """

    if at < 0 and take + at <= 0:
        return slice(at, None)

    if take < 0:
        return slice(at + take, at)

    return slice(at, at + take)
