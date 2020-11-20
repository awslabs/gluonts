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

from typing import Iterable, Iterator, Callable, Optional, List
import itertools
import random
import logging
import multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
import io
import pickle
import sys
from queue import Empty

from gluonts.dataset.common import DataEntry, DataBatch, Dataset
from gluonts.dataset.util import MPWorkerInfo
from gluonts.itertools import batcher, cyclic, pseudo_shuffled
from gluonts.transform import Transformation, TransformedDataset

logger = logging.getLogger(__name__)


def construct_training_iterator(
    dataset: Dataset,
    *,
    transform: Transformation,
    shuffle_buffer_length: Optional[int] = None,
) -> Iterator[DataEntry]:
    transformed_dataset = TransformedDataset(
        cyclic(dataset), transform, is_train=True,
    )

    if shuffle_buffer_length is None:
        return iter(transformed_dataset)
    else:
        return pseudo_shuffled(
            iter(transformed_dataset),
            shuffle_buffer_length=shuffle_buffer_length,
        )


class MultiProcessBatcher(Iterator):
    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        batch_size: int,
        stack_fn: Callable,
        num_workers: int,
        max_queue_size: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        decode_fn: Callable = lambda x: x,
    ):
        assert num_workers >= 1
        assert max_queue_size is None or max_queue_size >= num_workers

        self.batch_size = batch_size
        self.stack_fn = stack_fn
        self.decode_fn = decode_fn
        self.num_workers = num_workers
        self.max_queue_size = (
            max_queue_size if max_queue_size is not None else 5 * num_workers
        )

        self.manager = mp.Manager()
        self.batch_queue = self.manager.Queue(maxsize=self.max_queue_size)
        self.terminate_event = self.manager.Event()
        self.exhausted_events = [
            self.manager.Event() for _ in range(self.num_workers)
        ]
        self.processes = []

        for worker_id, event in enumerate(self.exhausted_events):
            p = mp.Process(
                target=self.worker_fn,
                args=(
                    worker_id,
                    num_workers,
                    dataset,
                    transform,
                    shuffle_buffer_length,
                    batch_size,
                    stack_fn,
                    self.batch_queue,
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
        dataset,
        transform,
        shuffle_buffer_length: int,
        batch_size: int,
        stack_fn: Callable,
        batch_queue: mp.Queue,
        terminate_event,
        exhausted_event,
    ):
        MPWorkerInfo.set_worker_info(
            num_workers=num_workers, worker_id=worker_id,
        )

        data_iterator = construct_training_iterator(
            dataset,
            transform=transform,
            shuffle_buffer_length=shuffle_buffer_length,
        )

        for batch in batcher(data_iterator, batch_size):
            stacked_batch = stack_fn(batch)
            try:
                if terminate_event.is_set():
                    return
                buf = io.BytesIO()
                ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(
                    (worker_id, stacked_batch)
                )
                batch_queue.put(buf.getvalue())
            except (EOFError, BrokenPipeError):
                return

        exhausted_event.set()

    def __iter__(self):
        return self

    def __next__(self):
        if (
            all(event.is_set() for event in self.exhausted_events)
            and self.batch_queue.empty()
        ):
            self._halt_processes()
            raise StopIteration

        try:
            # TODO make timeout configurable
            got = self.batch_queue.get(timeout=120)
            worker_id, batch = pickle.loads(got)
            batch = self.decode_fn(batch)
        except Empty:
            raise StopIteration()

        return batch

    def _empty_queue(self):
        try:
            batch = self.batch_queue.get(block=False)
            while batch:
                self.batch_queue.get(block=False)
        except (Empty, FileNotFoundError):
            pass

    def _halt_processes(self):
        # Send termination message to workers
        self.terminate_event.set()
        # Empty queue to make sure workers get the message
        self._empty_queue()
        for p in self.processes:
            p.join()


class DataLoader(Iterable[DataBatch]):
    pass


def win32_guard(num_workers: Optional[int]) -> Optional[int]:
    if num_workers and sys.platform == "win32":
        logger.warning(
            "Multiprocessing is not supported on Windows, "
            "num_workers will be set to None."
        )
        return None
    return num_workers


class TrainDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        stack_fn: Callable,
        num_batches_per_epoch: int,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        decode_fn: Callable = lambda x: x,
    ) -> None:
        self.batch_size = batch_size
        self.stack_fn = stack_fn
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = win32_guard(num_workers)
        self.num_prefetch = num_prefetch
        self.shuffle_buffer_length = shuffle_buffer_length

        if not self.num_workers:
            iterator = construct_training_iterator(
                dataset,
                transform=transform,
                shuffle_buffer_length=shuffle_buffer_length,
            )
            self.batch_iterator = map(stack_fn, batcher(iterator, batch_size))
        else:
            self.batch_iterator = MultiProcessBatcher(
                dataset,
                transform=transform,
                batch_size=batch_size,
                stack_fn=stack_fn,
                decode_fn=decode_fn,
                num_workers=self.num_workers,
                max_queue_size=num_prefetch,
                shuffle_buffer_length=shuffle_buffer_length,
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
        stack_fn: Callable,
        # FIXME: the following aren't used
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
    ) -> None:
        self.transformed_dataset = TransformedDataset(
            dataset, transform, is_train=True,
        )
        self.batch_size = batch_size
        self.stack_fn = stack_fn

    def __iter__(self):
        yield from map(
            self.stack_fn, batcher(self.transformed_dataset, self.batch_size),
        )


class InferenceDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        stack_fn: Callable,
        # FIXME: the following aren't used
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
    ) -> None:
        self.transformed_dataset = TransformedDataset(
            dataset, transform, is_train=False,
        )
        self.batch_size = batch_size
        self.stack_fn = stack_fn

    def __iter__(self):
        yield from map(
            self.stack_fn, batcher(self.transformed_dataset, self.batch_size),
        )
