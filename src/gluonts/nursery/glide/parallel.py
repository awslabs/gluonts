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
from functools import partial

from toolz.itertoolz import partition_all as into_batches
from toolz.functoolz import identity


class sentinel:
    pass


def map_to_queue(fn, emitter, queue, encode, batch_size):
    for batch in into_batches(batch_size, map(encode, fn(emitter))):
        queue.put(batch)
    queue.put(sentinel)


class ParApplyIterator:
    def __init__(self, procs, queue, decode):
        self.procs = procs
        self.queue = queue
        self.decode = decode
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
        return self.decode(self._current.pop())


class ParApply:
    def __init__(
        self,
        fn,
        emitter,
        batch_size=1,
        queue_size=50,
        encode=identity,
        decode=identity,
    ):
        self.fn = fn
        self.emitter = emitter
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.encode = encode
        self.decode = decode

    def __iter__(self):
        queue = mp.Queue(self.queue_size)

        Process = partial(mp.Process, target=map_to_queue, daemon=True)

        it = ParApplyIterator(
            procs=[
                Process(
                    args=(
                        self.fn,
                        emitter,
                        queue,
                        self.encode,
                        self.batch_size,
                    )
                )
                for emitter in self.emitter
            ],
            queue=queue,
            decode=self.decode,
        )
        it.start()
        return it
