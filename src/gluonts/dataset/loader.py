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

import io
import itertools
import logging
import multiprocessing as mp
import pickle
import random
import sys
from multiprocessing.reduction import ForkingPickler
from queue import Empty
from typing import Callable, Iterable, Iterator, List, Optional

from gluonts.dataset.common import DataBatch, DataEntry, Dataset
from gluonts.dataset.util import MPWorkerInfo
from gluonts.itertools import batcher, Cyclic, IterableSlice, PseudoShuffled
from gluonts.transform import Transformation, TransformedDataset

logger = logging.getLogger(__name__)


class MultiProcessBatcher(Iterator):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        stack_fn: Callable,
        num_workers: int,
        max_queue_size: Optional[int] = None,
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
        batch_size: int,
        stack_fn: Callable,
        batch_queue: mp.Queue,
        terminate_event,
        exhausted_event,
    ):
        MPWorkerInfo.set_worker_info(
            num_workers=num_workers,
            worker_id=worker_id,
        )

        for batch in batcher(dataset, batch_size):
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


def win32_guard(num_workers: Optional[int]) -> Optional[int]:
    if num_workers and sys.platform == "win32":
        logger.warning(
            "Multiprocessing is not supported on Windows, "
            "num_workers will be set to None."
        )
        return None
    return num_workers


class DataLoader(Iterable[DataBatch]):
    def __init__(
        self,
        data_iterable: Iterable[DataEntry],
        *,
        batch_size: int,
        stack_fn: Callable,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        decode_fn: Callable = lambda x: x,
    ) -> None:
        self.data_iterable = data_iterable
        self.batch_size = batch_size
        self.stack_fn = stack_fn
        self.num_workers = win32_guard(num_workers)
        self.num_prefetch = num_prefetch
        self.decode_fn = decode_fn

    def __iter__(self):
        batch_iterator = (
            map(self.stack_fn, batcher(self.data_iterable, self.batch_size))
            if not self.num_workers
            else MultiProcessBatcher(
                self.data_iterable,
                batch_size=self.batch_size,
                stack_fn=self.stack_fn,
                decode_fn=self.decode_fn,
                num_workers=self.num_workers,
                max_queue_size=self.num_prefetch,
            )
        )

        return batch_iterator


# TODO: the following are for backward compatibility, and could eventually be removed


def TrainDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation,
    batch_size: int,
    stack_fn: Callable,
    num_batches_per_epoch: Optional[int] = None,
    num_workers: Optional[int] = None,
    num_prefetch: Optional[int] = None,
    shuffle_buffer_length: Optional[int] = None,
    decode_fn: Callable = lambda x: x,
):
    transformed_dataset = TransformedDataset(
        Cyclic(dataset), transform, is_train=True
    )
    data_iterable = (
        PseudoShuffled(
            transformed_dataset, shuffle_buffer_length=shuffle_buffer_length
        )
        if shuffle_buffer_length is not None
        else transformed_dataset
    )
    data_loader = DataLoader(
        data_iterable=data_iterable,
        batch_size=batch_size,
        stack_fn=stack_fn,
        num_workers=num_workers,
        num_prefetch=num_prefetch,
        decode_fn=decode_fn,
    )
    return (
        iter(data_loader)
        if num_batches_per_epoch is None
        else IterableSlice(iter(data_loader), num_batches_per_epoch)
    )


def ValidationDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation,
    batch_size: int,
    stack_fn: Callable,
    num_workers: Optional[int] = None,
    num_prefetch: Optional[int] = None,
    shuffle_buffer_length: Optional[int] = None,
):
    return DataLoader(
        data_iterable=TransformedDataset(dataset, transform, is_train=True),
        batch_size=batch_size,
        stack_fn=stack_fn,
    )


def InferenceDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation,
    batch_size: int,
    stack_fn: Callable,
    num_workers: Optional[int] = None,
    num_prefetch: Optional[int] = None,
    shuffle_buffer_length: Optional[int] = None,
):
    return DataLoader(
        data_iterable=TransformedDataset(dataset, transform, is_train=False),
        batch_size=batch_size,
        stack_fn=stack_fn,
    )
