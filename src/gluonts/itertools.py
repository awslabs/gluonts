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

import itertools
import math
import pickle
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Sequence,
    Tuple,
)
from typing_extensions import Protocol, runtime_checkable

from toolz import curry


@runtime_checkable
class SizedIterable(Protocol):
    def __len__(self):
        ...

    def __iter__(self):
        ...


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
class Cached:
    """
    An iterable wrapper, which caches values in a list the first time it is
    iterated.

    The primary use-case for this is to avoid re-computing the element of the
    sequence, in case the inner iterable does it on demand.

    This should be used to wrap deterministic iterables, i.e. iterables where
    the data generation process is not random, and that yield the same
    elements when iterated multiple times.
    """

    iterable: SizedIterable
    cache: list = field(default_factory=list, init=False)

    def __iter__(self):
        if not self.cache:
            for element in self.iterable:
                yield element
                self.cache.append(element)
        else:
            yield from self.cache

    def __len__(self) -> int:
        return len(self.iterable)


@dataclass
class PickleCached:
    """A caching wrapper for ``iterable`` using ``pickle`` to store cached
    values on disk.

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
    """Get attribute ``name`` from elements in ``seq``."""

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
