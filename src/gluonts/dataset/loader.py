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
from multiprocessing.reduction import ForkingPickler
from typing import Callable, Iterable, Optional

from pydantic import BaseModel

from gluonts.dataset.common import DataBatch, Dataset
from gluonts.dataset.util import MPWorkerInfo
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled, batcher
from gluonts.transform import AdhocTransform, Identity, Transformation

logger = logging.getLogger(__name__)


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


def _encode(value):
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(value)
    return buf.getvalue()


def worker_fn(
    worker_id: int,
    dataset,
    num_workers: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    MPWorkerInfo.set_worker_info(
        num_workers=num_workers,
        worker_id=worker_id,
    )

    while True:
        try:
            input_queue.get()
            for encoded_entry in map(_encode, dataset):
                output_queue.put(encoded_entry)
            output_queue.put(_encode(None))
        except (EOFError, BrokenPipeError):
            return


DataLoader = Iterable[DataBatch]


class MultiProcessLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        num_workers: int,
        max_queue_size: Optional[int] = None,
        decode_fn: Callable = lambda x: x,
        queue_timeout_seconds: int = 120,
    ):
        assert num_workers >= 1

        if max_queue_size is None:
            max_queue_size = 5 * num_workers
        else:
            assert max_queue_size >= num_workers

        self.decode_fn = decode_fn
        self.queue_timeout_seconds = queue_timeout_seconds
        self.manager = mp.Manager()
        self.output_queue = self.manager.Queue(maxsize=max_queue_size)
        self.input_queues = [self.manager.Queue() for _ in range(num_workers)]
        self.num_workers = num_workers

        self.processes = [
            mp.Process(
                target=worker_fn,
                kwargs={
                    "worker_id": worker_id,
                    "dataset": dataset,
                    "num_workers": num_workers,
                    "input_queue": input_queue,
                    "output_queue": self.output_queue,
                },
            )
            for worker_id, input_queue in enumerate(self.input_queues)
        ]

        for process in self.processes:
            process.start()

    def __iter__(self):
        num_finished = 0
        for input_queue in self.input_queues:
            input_queue.put(_encode(True))
        while num_finished < self.num_workers:
            raw = self.output_queue.get(timeout=self.queue_timeout_seconds)
            data = pickle.loads(raw)
            if data is None:
                num_finished += 1
                continue
            yield self.decode_fn(data)


# TODO: the following are for backward compatibility
# and could eventually be removed


class Batch(Transformation, BaseModel):
    batch_size: int

    def __call__(self, data, is_train):
        yield from batcher(data, self.batch_size)


def TrainDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
    num_batches_per_epoch: Optional[int] = None,
    num_prefetch: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle_buffer_length: Optional[int] = None,
    decode_fn: Callable = lambda x: x,
):
    """
    Construct an iterator of batches for training purposes.

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
    dataset: Dataset = Cyclic(dataset)

    if shuffle_buffer_length:
        dataset = PseudoShuffled(dataset, shuffle_buffer_length)

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    transformed_dataset = transform.apply(dataset, is_train=True)

    if num_workers is not None:
        loader = MultiProcessLoader(
            transformed_dataset,
            decode_fn=decode_fn,
            num_workers=num_workers,
            max_queue_size=num_prefetch,
        )
        batches = iter(loader)
    else:
        batches = iter(transformed_dataset)

    if num_batches_per_epoch is None:
        return batches
    else:
        return IterableSlice(batches, num_batches_per_epoch)


def ValidationDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
    num_prefetch: Optional[int] = None,
    num_workers: Optional[int] = None,
    decode_fn: Callable = lambda x: x,
):
    """
    Construct an iterator of batches for validation purposes.

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
    num_workers
        Number of worker processes to use. Default: None.
    num_prefetch
        Sets the length of the queue of batches being produced by worker
        processes. (Only meaningful when ``num_workers is not None``).
    decode_fn
        A function called on each batch after it's been taken out of the queue.
        (Only meaningful when ``num_workers is not None``).

    Returns
    -------
    Iterable[DataBatch]
        An iterable sequence of batches.
    """

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    transformed_dataset = transform.apply(dataset, is_train=True)

    if num_workers is None:
        return transformed_dataset

    return MultiProcessLoader(
        transformed_dataset,
        decode_fn=decode_fn,
        num_workers=num_workers,
        max_queue_size=num_prefetch,
    )


def InferenceDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
):
    """
    Construct an iterator of batches for inference purposes.

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
