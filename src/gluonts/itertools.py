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
import random
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
from dataclasses import dataclass, field

from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class SizedIterable(Protocol):
    def __len__(self):
        ...

    def __iter__(self):
        ...


T = TypeVar("T")


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


class Map:
    def __init__(self, fn, iterable: SizedIterable):
        self.fn = fn
        self.iterable = iterable

    def __iter__(self):
        return map(self.fn, self.iterable)

    def __len__(self):
        return len(self.iterable)

    def __repr__(self):
        return f"Map(data={self.iterable!r})"


K = TypeVar("K")
V = TypeVar("V")


def rows_to_columns(
    rows: Sequence[Dict[K, V]],
    wrap: Callable[[Sequence[V]], Sequence[V]] = lambda x: x,
) -> Dict[K, Sequence[V]]:
    """Transpose rows of dicts, to one dict containing columns.

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
    """Transpose column-orientation to row-orientation.

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
    """`roundrobin('ABC', 'D', 'EF') --> A D E B F C`

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
    it: Iterator[T], fn: Callable[[T], bool]
) -> Tuple[List[T], List[T]]:
    """Partition `it` into two lists given predicate `fn`.

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
