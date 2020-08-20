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

import multiprocessing as mp
from typing import Callable, Iterator, TypeVar, Generic

from toolz.itertoolz import concat
from toolz.itertoolz import partition_all as batcher

from ._partition import partition

T = TypeVar("T")
U = TypeVar("U")


class sentinel:
    pass


class ParMapper(Generic[T, U]):
    def __init__(
        self,
        fn: Callable[[T], U],
        emitter: Iterator[T],
        queue: mp.Queue,
        batch_size: int = 8,
    ):
        self.fn = fn
        self.emitter = emitter
        self.queue = queue
        self.batch_size = batch_size

    def __call__(self) -> None:
        stream = map(self.fn, self.emitter)
        for batch in batcher(self.batch_size, stream):
            self.queue.put(batch)
        self.queue.put(sentinel)


def map_to_queue(fn, emitter, queue, batch_size):
    stream = map(fn, emitter)
    for batch in batcher(batch_size, stream):
        queue.put(batch)
    queue.put(sentinel)


class Map:
    def __init__(self, fn, partitions: list):
        self.fn = fn
        self.partitions = partitions

    def __iter__(self):
        yield from concat(
            map(self.fn, partition) for partition in self.partitions
        )


class ParPipelineIterator:
    def __init__(self, procs, queue):
        self.procs = procs
        self.queue = queue
        self._current = []
        self._sentinel_count = 0

    def join(self):
        for proc in self.procs:
            proc.join()

    def start(self):
        for proc in self.procs:
            proc.start()

    def kill(self):
        for proc in self.procs:
            proc.kill()

    def terminate(self):
        for proc in self.procs:
            proc.terminate()

    def __next__(self):
        while not self._current:
            val = self.queue.get()
            if val is sentinel:
                self._sentinel_count += 1
                if self._sentinel_count >= len(self.procs):
                    raise StopIteration
            else:
                self._current = list(reversed(val))

        return self._current.pop()


class ParMap:
    def __init__(self, fn, emitter, batch_size=128):
        self.fn = fn
        self.emitter = emitter
        self.batch_size = batch_size

    def __iter__(self):
        queue = mp.Queue()

        it = ParPipelineIterator(
            [
                mp.Process(
                    target=map_to_queue,
                    args=(self.fn, emitter, queue, self.batch_size),
                    daemon=True,
                )
                for emitter in self.emitter
            ],
            queue,
        )
        it.start()
        return it
