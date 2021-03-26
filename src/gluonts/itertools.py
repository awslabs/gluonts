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
from typing import Iterable, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class Cyclic(Iterable):
    """
    Like `itertools.cycle`, but does not store the data.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable

    def __iter__(self):
        at_least_one = False
        while True:
            for el in self.iterable:
                at_least_one = True
                yield el
            if not at_least_one:
                break


def batcher(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Groups elements from `iterable` into batches of size `batch_size`.

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


class Cached(Iterable):
    """
    An iterable wrapper, which caches values in a list the first time it is iterated.

    The primary use-case for this is to avoid re-computing the element of the sequence,
    in case the inner iterable does it on demand.

    This should be used to wrap deterministic iterables, i.e. iterables where the data
    generation process is not random, and that yield the same elements when iterated
    multiple times.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.cache = None

    def __iter__(self):
        if self.cache is None:
            self.cache = []
            for element in self.iterable:
                yield element
                self.cache.append(element)
        else:
            yield from self.cache


class PseudoShuffled(Iterable):
    """
    Yields items from a given iterable in a pseudo-shuffled order.
    """

    def __init__(self, iterable: Iterable, shuffle_buffer_length: int) -> None:
        self.iterable = iterable
        self.shuffle_buffer_length = shuffle_buffer_length

    def __iter__(self):
        shuffle_buffer = []

        for element in self.iterable:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                yield shuffle_buffer.pop(random.randrange(len(shuffle_buffer)))

        while shuffle_buffer:
            yield shuffle_buffer.pop(random.randrange(len(shuffle_buffer)))


class IterableSlice(Iterable):
    """
    An iterable version of `itertools.islice`, i.e. one that can be iterated
    over multiple times.
    """

    def __init__(self, iterable: Iterable, length: Optional[int]) -> None:
        self.iterable = iterable
        self.length = length

    def __iter__(self):
        return itertools.islice(self.iterable, self.length)
