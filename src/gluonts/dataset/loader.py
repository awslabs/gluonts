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

from typing import Iterable, Iterator, Callable, Optional
import itertools
import random
from multiprocessing import Process, Manager, Queue
from multiprocessing.reduction import ForkingPickler
import io
import pickle
from queue import Empty

from gluonts.dataset.common import DataEntry, DataBatch, Dataset
from gluonts.dataset.util import MPWorkerInfo, batcher
from gluonts.transform import Transformation
from gluonts.transform.dataset import TransformedDataset


class CyclicIterable(Iterable):
    def __init__(self, base_iterable: Iterable) -> None:
        self.base_iterable = base_iterable

    def __iter__(self):
        while True:
            yield from self.base_iterable


class PseudoShuffledIterator(Iterator):
    """
    A wrapper class which takes a serialized iterator as an input and generates a
    pseudo randomized iterator using the same elements from the input iterator.
    """

    def __init__(self, base_iterator: Iterator, shuffle_buffer_length: int):
        self.shuffle_buffer: list = []
        self.shuffle_buffer_length = shuffle_buffer_length
        self.base_iterator = base_iterator
        self.base_iter_finished = False

    def __next__(self):
        # if the buffer is empty, fill the buffer first.
        # (Should only executed in the first round)
        if not self.shuffle_buffer:
            self.shuffle_buffer = list(
                itertools.islice(
                    self.base_iterator, self.shuffle_buffer_length
                )
            )

        # If buffer still empty, means all elements used, return a signal of
        # end of iterator
        if not self.shuffle_buffer:
            raise StopIteration

        # Choose an element at a random index and yield it and fill it with
        # the next element in the sequential generator
        idx = random.randint(0, len(self.shuffle_buffer) - 1)
        next_sample = self.shuffle_buffer[idx]

        # Replace the index with the next element in the iterator if the
        # iterator has not finished. Delete the index otherwise.
        try:
            self.shuffle_buffer[idx] = next(self.base_iterator)
        except StopIteration:
            del self.shuffle_buffer[idx]

        return next_sample


class MultiProcessBatcher(Iterator):
    def __init__(
        self,
        base_iterable: Iterable,
        batch_size: int,
        batchify_fn: Callable,
        num_workers: int,
        max_queue_size: Optional[int] = None,
    ):
        assert num_workers >= 1
        assert max_queue_size is None or max_queue_size >= num_workers

        self.base_iterable = base_iterable
        self.batch_size = batch_size
        self.batchify_fn = batchify_fn
        self.num_workers = num_workers
        self.max_queue_size = (
            max_queue_size if max_queue_size is not None else 5 * num_workers
        )

        self.manager = Manager()
        self.data_queue = self.manager.Queue(maxsize=self.max_queue_size)
        self.terminate_event = self.manager.Event()
        self.exhausted_events = [
            self.manager.Event() for _ in range(self.num_workers)
        ]
        self.processes = []

        for wid, event in enumerate(self.exhausted_events):
            p = Process(
                target=self.worker_fn,
                args=(
                    wid,
                    self.num_workers,
                    self.base_iterable,
                    self.batch_size,
                    self.batchify_fn,
                    self.data_queue,
                    self.terminate_event,
                    event,
                ),
            )
            p.start()
            self.processes.append(p)

        self.count = 0

    @staticmethod
    def worker_fn(
        worker_id: int,
        num_workers: int,
        iterable: Iterable,
        batch_size: int,
        batchify_fn: Callable,
        data_queue: Queue,
        terminate_event,
        exhausted_event,
    ):
        MPWorkerInfo.worker_process = True
        MPWorkerInfo.worker_id = worker_id
        MPWorkerInfo.num_workers = num_workers

        for batch_list in batcher(iterable, batch_size):
            batch = batchify_fn(batch_list)
            try:
                if terminate_event.is_set():
                    return
                buf = io.BytesIO()
                ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(
                    (worker_id, batch)
                )
                data_queue.put(buf.getvalue())
            except (EOFError, BrokenPipeError):
                return

        exhausted_event.set()

    def __iter__(self):
        return self

    def __next__(self):
        if (
            all(event.is_set() for event in self.exhausted_events)
            and self.data_queue.empty()
        ):
            self._halt_processes()
            raise StopIteration()

        try:
            # TODO make timeout configurable
            got = self.data_queue.get(timeout=120)
            worker_id, batch = pickle.loads(got)
        except Empty:
            raise StopIteration()

        return batch

    def _empty_queue(self):
        try:
            batch = self.data_queue.get(block=False)
            while batch:
                self.data_queue.get(block=False)
        except (Empty, FileNotFoundError):
            pass

    def _halt_processes(self):
        try:
            # Send termination message to workers
            self.terminate_event.set()
        except FileNotFoundError:
            pass
        # Empty queue to make sure workers get the message
        self._empty_queue()
        for p in self.processes:
            p.join()

    def __del__(self):
        self._halt_processes()


class DataLoader(Iterable[DataBatch]):
    pass


class TrainDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        batchify_fn: Callable,
        num_batches_per_epoch: int,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.batchify_fn = batchify_fn
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = num_workers
        self.num_prefetch = num_prefetch
        self.shuffle_buffer_length = shuffle_buffer_length

        transformed_dataset = TransformedDataset(
            base_dataset=CyclicIterable(dataset),
            transformation=transform,
            is_train=True,
        )

        shuffled_iterator: Iterable[DataEntry] = (
            iter(transformed_dataset)
            if shuffle_buffer_length is None
            else PseudoShuffledIterator(
                iter(transformed_dataset),
                shuffle_buffer_length=shuffle_buffer_length,
            )
        )

        self.batch_iterator = (
            map(batchify_fn, batcher(shuffled_iterator, batch_size))
            if num_workers is None
            else MultiProcessBatcher(
                shuffled_iterator,
                batch_size=batch_size,
                batchify_fn=batchify_fn,
                num_workers=num_workers,
                max_queue_size=num_prefetch,
            )
        )

    def __len__(self):
        return self.num_batches_per_epoch

    def __iter__(self):
        yield from itertools.islice(
            self.batch_iterator, self.num_batches_per_epoch
        )


class ValidationDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        batchify_fn: Callable,
        # FIXME: the following aren't used
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
    ) -> None:
        self.transformed_dataset = TransformedDataset(
            base_dataset=dataset, transformation=transform, is_train=True,
        )
        self.batch_size = batch_size
        self.batchify_fn = batchify_fn

    def __iter__(self):
        yield from map(
            self.batchify_fn,
            batcher(self.transformed_dataset, self.batch_size),
        )


class InferenceDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        batchify_fn: Callable,
        # FIXME: the following aren't used
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
    ) -> None:
        self.transformed_dataset = TransformedDataset(
            base_dataset=dataset, transformation=transform, is_train=False,
        )
        self.batch_size = batch_size
        self.batchify_fn = batchify_fn

    def __iter__(self):
        yield from map(
            self.batchify_fn,
            batcher(self.transformed_dataset, self.batch_size),
        )
