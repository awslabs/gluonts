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

from gluonts.dataset.common import DataBatch, DataEntry, Dataset
from gluonts.dataset.util import MPWorkerInfo
from gluonts.itertools import batcher, Cyclic, IterableSlice, PseudoShuffled
from gluonts.transform import Transformation

logger = logging.getLogger(__name__)


def _encode(value):
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(value)
    return buf.getvalue()


def worker_fn(
    worker_id: int,
    num_workers: int,
    dataset,
    batch_size: int,
    stack_fn: Callable,
    batch_queue: mp.Queue,
):
    MPWorkerInfo.set_worker_info(num_workers=num_workers, worker_id=worker_id)

    batches = batcher(dataset, batch_size)
    batches = map(stack_fn, batches)
    raw_batches = map(_encode, batches)

    for raw in raw_batches:
        try:
            batch_queue.put(raw)
        except (EOFError, BrokenPipeError):
            return


class MultiProcessBatcher(Iterable):
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

        if max_queue_size is None:
            max_queue_size = 5 * num_workers
        else:
            assert max_queue_size >= num_workers

        self.decode_fn = decode_fn
        self.batch_queue = mp.Manager().Queue(maxsize=max_queue_size)

        common_kwargs = {
            "num_workers": num_workers,
            "dataset": dataset,
            "batch_size": batch_size,
            "stack_fn": stack_fn,
            "batch_queue": self.batch_queue,
        }
        create_worker = partial(
            mp.Process, target=worker_fn, kwargs=common_kwargs
        )

        self.processes = [
            create_worker(args=[worker_id]) for worker_id in range(num_workers)
        ]
        for process in self.processes:
            process.start()

    def _has_values(self):
        alive = any(proc.is_alive() for proc in self.processes)
        return alive or not self.batch_queue.empty()

    def __iter__(self):
        while self._has_values():
            yield self._get()

        self._terminate()

    def _get(self):
        # TODO make timeout configurable
        raw = self.batch_queue.get(timeout=120)
        return self.decode_fn(pickle.loads(raw))

    def _terminate(self):
        for process in self.processes:
            process.terminate()


def win32_guard(num_workers: Optional[int]) -> Optional[int]:
    if num_workers and sys.platform == "win32":
        logger.warning(
            "Multiprocessing is not supported on Windows, "
            "num_workers will be set to None."
        )
        return None
    return num_workers


class DataLoader(Iterable[DataBatch]):
    """Iterate a datasets and stack its entries into batches of a given size.

    The object can be configured to use multiple processes to iterate the
    entries, which increases throughput in case the data entries are lazily
    transformed.

    Parameters
    ----------
    data_iterable
        Data to construct batches from.
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
    """

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
        assert num_workers is None or num_workers > 0
        assert num_prefetch is None or num_prefetch > 0

        self.data_iterable = data_iterable
        self.batch_size = batch_size
        self.stack_fn = stack_fn
        self.num_workers = win32_guard(num_workers)
        self.num_prefetch = num_prefetch
        self.decode_fn = decode_fn

    def __iter__(self):
        if self.num_workers is None:
            return map(
                self.stack_fn, batcher(self.data_iterable, self.batch_size)
            )

        return iter(
            MultiProcessBatcher(
                self.data_iterable,
                batch_size=self.batch_size,
                stack_fn=self.stack_fn,
                decode_fn=self.decode_fn,
                num_workers=self.num_workers,
                max_queue_size=self.num_prefetch,
            )
        )


# TODO: the following are for backward compatibility
# and could eventually be removed


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
    transformed_dataset = transform.apply(Cyclic(dataset))
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
    return DataLoader(
        data_iterable=transform.apply(dataset),
        batch_size=batch_size,
        stack_fn=stack_fn,
    )


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
    return DataLoader(
        data_iterable=transform.apply(dataset, is_train=False),
        batch_size=batch_size,
        stack_fn=stack_fn,
    )
