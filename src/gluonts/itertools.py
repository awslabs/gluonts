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

from typing import Iterable, Iterator, List, TypeVar
import itertools
import random

T = TypeVar("T")


def cycle(it):
    """Like `itertools.cycle`, but does not store the data."""

    while True:
        yield from it


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


class cache(Iterable):
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


class pseudo_shuffle(Iterator):
    """
    An iterator that yields item from a wrapped iterator in a pseudo-shuffled order.
    """

    def __init__(self, iterator: Iterator, shuffle_buffer_length: int):
        self.shuffle_buffer: list = []
        self.shuffle_buffer_length = shuffle_buffer_length
        self.iterator = iterator

    def __next__(self):
        # If the buffer is empty, fill the buffer first.
        if not self.shuffle_buffer:
            self.shuffle_buffer = list(
                itertools.islice(self.iterator, self.shuffle_buffer_length)
            )

        # If buffer still empty, means all elements used, return a signal of
        # end of iterator
        if not self.shuffle_buffer:
            raise StopIteration

        # Choose an element at a random index and yield it and fill it with
        # the next element in the sequential generator
        idx = random.randrange(len(self.shuffle_buffer))
        next_sample = self.shuffle_buffer[idx]

        # Replace the index with the next element in the iterator if the
        # iterator has not finished. Delete the index otherwise.
        try:
            self.shuffle_buffer[idx] = next(self.iterator)
        except StopIteration:
            del self.shuffle_buffer[idx]

        return next_sample
