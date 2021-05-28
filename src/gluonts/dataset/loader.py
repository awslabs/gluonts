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
import logging
import multiprocessing as mp
import pickle
import sys
from functools import partial
from multiprocessing.reduction import ForkingPickler
from typing import Callable, Iterable, Iterator, Optional

from pydantic import BaseModel, validator
from toolz import compose_left

from gluonts.env import env
from gluonts.dataset.common import DataBatch, DataEntry, Dataset
from gluonts.itertools import batcher, Cyclic, IterableSlice, PseudoShuffled
from gluonts.transform import (
    Transformation,
    FlatMapTransformation,
    AdhocTransform,
)

logger = logging.getLogger(__name__)


class LoaderSettings(BaseModel):
    class Config:
        validate_assignment = True

    num_workers: Optional[int]
    num_prefetch: Optional[int]
    worker_id: Optional[int]

    @validator("num_workers")
    def win32_guard(cls, num_workers):
        if num_workers is None:
            return None

        assert num_workers > 0, "num_workers can't be negative"

        if sys.platform == "win32":
            logger.warning(
                "Multiprocessing is not supported on Windows, "
                "num_workers will be set to None."
            )

            return None

        return num_workers


env._declare("loader", LoaderSettings, default=LoaderSettings())


def _encode(value):
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(value)
    return buf.getvalue()


def worker_fn(
    worker_id: int,
    dataset,
    result_queue: mp.Queue,
):
    # env._push(num_workers=num_workers, worker_id=worker_id)
    env._push(loader={"worker_id": worker_id})

    for raw in map(_encode, dataset):
        try:
            result_queue.put(raw)
        except (EOFError, BrokenPipeError):
            return


class MultiProcessLoader(Iterable):
    @env._inject(
        num_workers="loader.num_workers", max_queue_size="loader.num_prefetch"
    )
    def __init__(
        self,
        dataset: Dataset,
        num_workers: int,
        max_queue_size: Optional[int],
        decode_fn: Callable = lambda x: x,
    ):
        # assert num_workers >= 1

        if max_queue_size is None:
            max_queue_size = 5 * num_workers
        else:
            assert max_queue_size >= num_workers

        self.decode_fn = decode_fn
        self.result_queue = mp.Manager().Queue(maxsize=max_queue_size)

        create_worker = partial(
            mp.Process,
            target=worker_fn,
            kwargs={
                "dataset": dataset,
                "result_queue": self.result_queue,
            },
        )

        self.processes = [
            create_worker(args=[worker_id]) for worker_id in range(num_workers)
        ]
        for process in self.processes:
            process.start()

    def _has_values(self):
        alive = any(proc.is_alive() for proc in self.processes)
        return alive or not self.result_queue.empty()

    def __iter__(self):
        while self._has_values():
            yield self._get()

        self._terminate()

    def _get(self):
        # TODO make timeout configurable
        raw = self.result_queue.get(timeout=120)
        return self.decode_fn(pickle.loads(raw))

    def _terminate(self):
        for process in self.processes:
            process.terminate()


DataLoader = Iterable[DataBatch]

# TODO: the following are for backward compatibility
# and could eventually be removed


class Batch(Transformation, BaseModel):
    batch_size: int

    def __call__(self, data, is_train) -> Iterator[DataEntry]:
        yield from batcher(data, self.batch_size)


def TrainDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation,
    batch_size: int,
    stack_fn: Callable,
    num_batches_per_epoch: Optional[int] = None,
    shuffle_buffer_length: Optional[int] = None,
    decode_fn: Callable = lambda x: x,
) -> Iterable[DataBatch]:
    """Construct an iterator of batches for training purposes.

    This function wraps around ``DataLoader`` to offer training-specific
    behaviour and options, as follows:

        1. The provided dataset is iterated cyclically, so that one can go over
        it multiple times in a single epoch. 2. A transformation must be
        provided, that is lazily applied as the dataset is being iterated;
        this is useful e.g. to slice random instances of fixed length out of
        each time series in the dataset. 3. The resulting batches can be
        iterated in a pseudo-shuffled order.

    The returned object is a stateful iterator, whose length is either
    ``num_batches_per_epoch`` (if not ``None``) or infinite (otherwise).

    Parameters
    ----------
    dataset
        Data to iterate over.
    transform
        Transformation to be lazily applied as data is being iterated.
        The transformation is applied in "training mode" (``is_train=True``).
    batch_size
        Number of entries to include in a batch.
    stack_fn
        Function to use to stack data entries into batches.
        This can be used to set a specific array type or computing device
        the arrays should end up onto (CPU, GPU).
    num_batches_per_epoch
        Length of the iterator. If ``None``, then the iterator is endless.
    num_workers
        Number of worker processes to use. Default: None.
    num_prefetch
        Sets the length of the queue of batches being produced by worker
        processes. (Only meaningful when ``num_workers is not None``).
    shuffle_buffer_length
        Size of the buffer used for shuffling. Default: None, in which case no
        shuffling occurs.
    decode_fn
        A function called on each batch after it's been taken out of the queue.
        (Only meaningful when ``num_workers is not None``).

    Returns
    -------
    Iterator[DataBatch]
        An iterator of batches.
    """
    dataset = Cyclic(dataset)

    if shuffle_buffer_length:
        dataset = PseudoShuffled(dataset, shuffle_buffer_length)

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    batches = transform.apply(dataset, is_train=True)

    if env.loader.num_workers is not None:
        batches = MultiProcessLoader(batches, decode_fn=decode_fn)

    batches = iter(batches)

    if num_batches_per_epoch is None:
        return batches
    else:
        return IterableSlice(batches, num_batches_per_epoch)


def ValidationDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation,
    batch_size: int,
    stack_fn: Callable,
):
    """Construct an iterator of batches for validation purposes.

    Parameters
    ----------
    dataset
        Data to iterate over.
    transform
        Transformation to be lazily applied as data is being iterated.
        The transformation is applied in "training mode" (``is_train=True``).
    batch_size
        Number of entries to include in a batch.
    stack_fn
        Function to use to stack data entries into batches.
        This can be used to set a specific array type or computing device
        the arrays should end up onto (CPU, GPU).

    Returns
    -------
    Iterable[DataBatch]
        An iterable sequence of batches.
    """

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    return transform.apply(dataset, is_train=True)


def InferenceDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation,
    batch_size: int,
    stack_fn: Callable,
):
    """Construct an iterator of batches for inference purposes.

    Parameters
    ----------
    dataset
        Data to iterate over.
    transform
        Transformation to be lazily applied as data is being iterated.
        The transformation is applied in "inference mode" (``is_train=False``).
    batch_size
        Number of entries to include in a batch.
    stack_fn
        Function to use to stack data entries into batches.
        This can be used to set a specific array type or computing device
        the arrays should end up onto (CPU, GPU).

    Returns
    -------
    Iterable[DataBatch]
        An iterable sequence of batches.
    """
    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    return transform.apply(dataset, is_train=False)
