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


# Standard library imports
import collections
import itertools
import logging
import pathlib
import pickle
import io
import random
import sys
import time
from collections import Sized
from multiprocessing.managers import SyncManager
from typing import Callable, Iterable, Optional, List, Iterator

import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
from multiprocessing.pool import Pool
from queue import Queue

try:
    import multiprocessing.resource_sharer
except ImportError:
    pass

# Third-party imports
import numpy as np
import pandas as pd
from mxnet import nd, context
import mxnet as mx

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import Dataset, DataEntry, DataBatch
from gluonts.transform import Transformation
from gluonts.dataset.util import MPWorkerInfo

# ForkingPickler related functions:
if sys.platform == "darwin" or sys.platform == "win32":

    def rebuild_ndarray(*args):
        """Rebuild ndarray from pickled shared memory"""
        # pylint: disable=no-value-for-parameter
        return nd.NDArray(nd.ndarray._new_from_shared_mem(*args))

    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        return rebuild_ndarray, data._to_shared_mem()


else:

    def rebuild_ndarray(pid, fd, shape, dtype):
        """Rebuild ndarray from pickled shared memory"""
        # pylint: disable=no-value-for-parameter
        fd = fd.detach()
        return nd.NDArray(
            nd.ndarray._new_from_shared_mem(pid, fd, shape, dtype)
        )

    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        # keep a local ref before duplicating fd
        data = data.as_in_context(context.Context("cpu_shared", 0))
        pid, fd, shape, dtype = data._to_shared_mem()
        fd = multiprocessing.reduction.DupFd(fd)
        return rebuild_ndarray, (pid, fd, shape, dtype)


ForkingPickler.register(nd.NDArray, reduce_ndarray)


def stack(
    data,
    multi_processing: bool,
    dtype: DType,
    single_process_ctx: Optional[mx.Context] = None,
):
    """Stack a list of data.
        Used when creating a single batch from list of dicts
        depending on whether multiprocessing is turned on, the batches will be
        constructed using different memory allocation techniques"""
    if isinstance(data[0], mx.nd.NDArray):
        if multi_processing:
            out = nd.empty(
                (len(data),) + data[0].shape,
                dtype=data[0].dtype,
                ctx=context.Context("cpu_shared", 0),
            )
            return mx.nd.stack(*data, out=out)
        else:
            return mx.nd.stack(*data)
    elif isinstance(data[0], np.ndarray):
        data = np.asarray(data)
        if data.dtype.kind == "f":
            data = data.astype(dtype)
        if multi_processing:
            # Workaround due to MXNet not being able to handle NDArrays with 0 in shape properly:
            if 0 in data.shape:
                return data
            return mx.nd.array(
                data, dtype=data.dtype, ctx=context.Context("cpu_shared", 0)
            )
        else:
            return mx.nd.array(data, dtype=data.dtype, ctx=single_process_ctx)
    elif isinstance(data[0], list):
        return list(
            stack(t, multi_processing, dtype, single_process_ctx)
            for t in zip(*data)
        )
    elif isinstance(data[0], tuple):
        return tuple(
            stack(t, multi_processing, dtype, single_process_ctx)
            for t in zip(*data)
        )
    elif isinstance(data[0], (pd.Timestamp, str, int, pathlib.PosixPath)):
        return data
    else:
        raise TypeError(
            f"Invalid type of data: {type(data[0])} for argument loss_function."
        )


def _batchify_fn(
    data: List[dict],
    dtype: DType,
    multi_processing: bool,
    single_process_ctx: Optional[mx.Context] = None,
) -> DataBatch:
    """reduce the list of dictionaries to a single dictionary, where values
        referenced by identical key are reduced using the stack function"""
    return {
        key: stack(
            data=[item[key] for item in data],
            multi_processing=multi_processing,
            dtype=dtype,
            single_process_ctx=single_process_ctx,
        )
        for key in data[0].keys()
    }


def _as_in_context(batch: dict, ctx: mx.Context) -> DataBatch:
    """Move data into new context, should only be in main process."""
    assert (
        not MPWorkerInfo.worker_process
    ), "This function is not meant to be used in workers."
    batch = {
        k: v.as_in_context(ctx) if isinstance(v, nd.NDArray)
        # Workaround due to MXNet not being able to handle NDArrays with 0 in shape properly:
        else (
            stack(v, False, v.dtype, ctx)
            if isinstance(v[0], np.ndarray) and 0 in v[0].shape
            else v
        )
        for k, v in batch.items()
    }
    return batch


# Each process has its own copy, so other processes can't interfere
class _WorkerData:
    """Contain the current data that the worker is using."""

    # dataset replica
    dataset: Optional[Dataset] = None
    # current dataset iterator in form of a transformation applied to the dataset
    transformation: Optional[Transformation] = None
    # replicate transformation
    dataset_iterator: Optional[Iterator[DataEntry]] = None
    # indicates which cycle the iterator has been reset last
    iterator_latest_reset_cycle: Optional[int] = 0
    # indicates whether the iterator was previously depleted
    iterator_exhausted_indicator: Optional[bool] = False
    # is used to cached transformed_samples in case  num_batches_for_shuffling > 1
    iterator_transformed_samples: Optional[Iterator[DataEntry]] = None
    # tracks how many batches have been retrieved from the
    iterator_transformed_samples_counter: Optional[int] = 0


def _worker_initializer(
    dataset: Dataset,
    transformation: Transformation,
    num_workers: int,
    worker_id_queue: Queue,
) -> None:
    """Initialier for processing pool."""

    _WorkerData.dataset = dataset
    _WorkerData.transformation = transformation

    # get unique worker id
    worker_id = int(worker_id_queue.get())
    multiprocessing.current_process().name = f"worker_{worker_id}"

    # propagate worker information
    MPWorkerInfo.set_worker_info(
        num_workers=num_workers, worker_id=worker_id, worker_process=True
    )


def _sequential_sample_generator(
    dataset, transformation, is_train, cyclic
) -> Iterator[DataEntry]:
    while True:
        for sample in transformation(data_it=dataset, is_train=is_train):
            yield sample
        # Dont cycle if not training time
        if not cyclic:
            return


def _worker_fn(
    batch_size: int,
    batchify_fn: Callable,
    dtype: DType,
    is_train: bool,
    shuffle: bool,
    num_batches_for_shuffling: int,
    cyclic: bool,
    cycle_num: int,
):
    """Function for processing data in worker process."""

    # initialize, or reset the iterator at each cycle
    assert isinstance(_WorkerData.iterator_latest_reset_cycle, int)
    if (_WorkerData.iterator_latest_reset_cycle < cycle_num) and (
        _WorkerData.iterator_latest_reset_cycle == 0 or not cyclic
    ):
        _worker_reset_iterator(is_train, cyclic, cycle_num)

    # retrieve the samples that will be batched
    batch_samples = None
    if num_batches_for_shuffling == 1:
        assert isinstance(
            _WorkerData.dataset_iterator, Iterable
        ), f"Dataset not Iterable: {type(_WorkerData.dataset_iterator)}."
        transformed_samples = list(
            itertools.islice(_WorkerData.dataset_iterator, batch_size)
        )
        if shuffle:
            random.shuffle(transformed_samples)
        batch_samples = transformed_samples
    elif num_batches_for_shuffling > 1:
        # if we haven't yet retrieved batches from the current num_batches_for_shuffling*batch_size samples chunk
        if _WorkerData.iterator_transformed_samples_counter == 0:
            assert isinstance(
                _WorkerData.dataset_iterator, Iterable
            ), f"Dataset not Iterable: {type(_WorkerData.dataset_iterator)}."
            transformed_samples = list(
                itertools.islice(
                    _WorkerData.dataset_iterator,
                    batch_size * num_batches_for_shuffling,
                )
            )
            random.shuffle(transformed_samples)
            _WorkerData.iterator_transformed_samples = iter(
                transformed_samples
            )
        assert isinstance(_WorkerData.iterator_transformed_samples, Iterable)
        batch_samples = list(
            itertools.islice(
                _WorkerData.iterator_transformed_samples, batch_size
            )
        )
        # drive the counter, and reset to 0 if all expected batches have been retrieved
        assert isinstance(
            _WorkerData.iterator_transformed_samples_counter, int
        )
        _WorkerData.iterator_transformed_samples_counter = (
            _WorkerData.iterator_transformed_samples_counter + 1
        ) % num_batches_for_shuffling
    else:
        raise AssertionError(
            f"Invalid value for num_batches_for_shuffling encountered: {num_batches_for_shuffling}."
        )

    # batch the samples, if there were any
    if batch_samples:
        success = True
        batch = batchify_fn(
            data=batch_samples, dtype=dtype, multi_processing=True
        )
    else:
        # the second time without being able to provide a batch we want to delay calling them again
        # on fist exhaustion they should not be delayed, since they need to indicate depletion
        # dont make the penalty to high, since that delays rescheduling of non empty iterators
        if _WorkerData.iterator_exhausted_indicator:
            time.sleep(0.05)
        else:
            _WorkerData.iterator_exhausted_indicator = True
        success = False
        batch = None

    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(
        (success, MPWorkerInfo.worker_id, batch)
    )
    return buf.getvalue()


# needed because some iterators are not cyclic
def _worker_reset_iterator(
    is_train: bool, cyclic: bool, cycle_num: int,
) -> None:
    """Initialize or reset iterators of workers."""

    _WorkerData.dataset_iterator = _sequential_sample_generator(
        dataset=_WorkerData.dataset,
        transformation=_WorkerData.transformation,
        is_train=is_train,
        cyclic=cyclic,
    )
    assert isinstance(_WorkerData.iterator_latest_reset_cycle, int)
    _WorkerData.iterator_latest_reset_cycle = cycle_num
    _WorkerData.iterator_exhausted_indicator = False


class _MultiWorkerIter(object):
    """Internal multi-worker iterator for DataLoader."""

    def __init__(
        self,
        worker_pool: Pool,
        batchify_fn: Callable,
        dtype: DType,
        ctx: mx.Context,
        is_train: bool,
        num_workers: int,
        batch_size: int,
        shuffle: bool,
        num_batches_for_shuffling: int,
        cyclic: bool,
        cycle_num: int,
        num_prefetch: int,
        worker_fn: Callable,
        dataset_len: int,
        timeout: int,
    ):
        self._worker_pool = worker_pool
        self._batchify_fn = batchify_fn
        self._data_buffer: dict = (
            {}
        )  # Its a dictionary with {request_id: data_batch} structure in our case
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._worker_fn = worker_fn
        self._timeout = timeout

        self._is_train = is_train
        self._dtype = dtype
        self._ctx = ctx
        self._cyclic = cyclic
        self._cycle_num = cycle_num
        # in case of cyclic=False iterators can be exhausted
        self._exhausted_iterators: set = set()
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_batches_for_shuffling = num_batches_for_shuffling
        self._dataset_len = dataset_len

        # pre-fetch batches
        self._num_prefetch = num_prefetch
        for i in range(self._num_prefetch):
            self._push_next()

    def __len__(self):
        return self._dataset_len

    def _push_next(self) -> None:
        """Assign next batch workload to workers."""
        # Optimally one would want to task worker that have none depleted iterators,
        # however, this does not seem to be possible with a worker pool
        async_ret = self._worker_pool.apply_async(
            self._worker_fn,
            (
                self._batch_size,
                self._batchify_fn,
                self._dtype,
                self._is_train,
                self._shuffle,
                self._num_batches_for_shuffling,
                self._cyclic,
                self._cycle_num,
            ),
        )
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def __next__(self) -> DataBatch:
        # Try to get a batch, sometimes its possible that an iterator was
        # exhausted and thus we don't get a new batch
        success = False
        while not success:
            try:
                self._push_next()

                if self._rcvd_idx == self._sent_idx:
                    assert (
                        not self._data_buffer
                    ), "Data buffer should be empty at this moment"
                    raise StopIteration
                assert (
                    self._rcvd_idx < self._sent_idx
                ), "rcvd_idx must be smaller than sent_idx"
                assert (
                    self._rcvd_idx in self._data_buffer
                ), "fatal error with _push_next, rcvd_idx missing"

                ret = self._data_buffer.pop(self._rcvd_idx)
                got = ret.get(self._timeout)
                self._rcvd_idx += 1

                # retrieve the batch from shared memory along with metadata
                success, worker_id, batch = pickle.loads(got)

                # If iterator exhausted/empty
                if not success:
                    self._exhausted_iterators.add(worker_id)
                    if self._num_workers == len(self._exhausted_iterators):
                        # No more batches to be generated
                        return {}
                    else:
                        self._push_next()
                else:
                    # either pin to cpu memory (with ctx=context.cpu_pinned(self.pin_device_id)),
                    # or return with the right context straight away
                    return _as_in_context(batch, self._ctx)
            except multiprocessing.context.TimeoutError:
                print(
                    f"Worker timed out after {self._timeout} seconds. This might be caused by "
                    "\n - Slow transform. Please increase timeout to allow slower data loading in each worker. "
                    "\n - Insufficient shared_memory if `timeout` is large enough. "
                    "\n Please consider to reduce `num_workers` or increase shared_memory in system."
                )
                raise
            except Exception:
                print("An unexpected error occurred in the WorkerIterator.")
                self._worker_pool.terminate()
                raise
        return {}

    def __iter__(self) -> Iterator[DataBatch]:
        while True:
            next_batch = next(self)
            if not next_batch:
                return
            yield next_batch

    def __del__(self):
        # Explicitly load the content from shared memory to delete it
        # Unfortunately it seems the way the data is pickled prevents efficient implicit GarbageCollection
        try:
            for k in list(self._data_buffer.keys()):
                res = pickle.loads(self._data_buffer.pop(k).get(self._timeout))
                del res
        except FileNotFoundError:
            # The resources were already released
            pass


class ParallelDataLoader(object):
    """
    Loads data from a dataset and returns mini-batches of data.

    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transformation
        A transformation to apply to each entry in the dataset.
    cyclic
        Whether the dataset in question should be cycled.
    is_train
        Whether the dataset in question is used for training.
    batch_size
        Size of mini-batch.
    ctx
        MXNet context to use to store data.
    dtype
        Floating point type to use.
    shuffle
        Whether to shuffle the samples.
    num_batches_for_shuffling
        The number of batches among which samples are shuffled. So for example if num_batches_for_shuffling = 8
        then the next num_batches_for_shuffling * 8 samples will be shuffled and then batched.
    num_workers
        The number of multiprocessing workers to use for data preprocessing.
        By default 0, in which case no multiprocessing will be utilized.
    num_prefetch
        The number of prefetching batches only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain batches before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more shared_memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_workers` in this case.
        By default it defaults to `num_workers * 2`.
    """

    def __init__(
        self,
        dataset: Dataset,
        transformation: Transformation,
        cyclic: bool,
        is_train: bool,
        batch_size: int,
        ctx: mx.Context,
        dtype: Optional[DType] = None,
        shuffle: Optional[bool] = None,
        num_batches_for_shuffling: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        num_workers: Optional[int] = None,
    ):
        # Some windows error with the ForkingPickler prevents usage currently:
        if sys.platform == "win32":
            logging.warning(
                "You have set `num_workers` to a non zero value, "
                "however, currently multiprocessing is not supported on windows and therefore"
                "`num_workers will be set to 0."
            )
            num_workers = 0
        assert (
            batch_size > 0
        ), "Batch size has to be a strictly positive integer."
        assert (
            num_batches_for_shuffling is None or num_batches_for_shuffling >= 1
        ), "Number of batches for shuffling has to be an integer >= 1."
        assert (
            num_workers is None or 0 <= num_workers
        ), "Num workers has to be >= 0."
        assert (
            num_prefetch is None or num_prefetch >= 0
        ), "Num workers has to be >= 0."

        self.dataset = dataset
        self.dataset_len: int
        if isinstance(dataset, Sized):
            assert isinstance(dataset, Sized)
            self.dataset_len = len(dataset)
        else:
            self.dataset_len = len(list(dataset))
        self.transformation = transformation
        # indicates that we want to cycle through the dataset
        self.cyclic = cyclic
        # indicates the current cycle, needed for resetting iterators at each cycle
        self.cycle_num = 0
        self.is_train = is_train
        self.batch_size = batch_size
        self.ctx = ctx

        self.dtype = dtype if dtype is not None else np.float32
        self.shuffle = shuffle if shuffle is not None else False
        self.num_batches_for_shuffling = (
            num_batches_for_shuffling
            if num_batches_for_shuffling is not None
            else 1
        )

        # TODO: switch to default multiprocessing.cpu_count() here
        default_num_workers = 0
        self.num_workers = (
            num_workers
            if num_workers is not None
            else min(
                self.dataset_len, default_num_workers
            )  # cannot have more than dataset entries
        )
        self.num_prefetch = (
            num_prefetch if num_prefetch is not None else 2 * self.num_workers
        )
        if self.num_prefetch < self.num_workers:
            logging.warning(
                "You have set `num_prefetch` to less than `num_workers`, which is counter productive."
                "If you want to reduce load, reduce `num_workers`."
            )
        self.worker_pool: Optional[Pool] = None
        self.worker_manager: Optional[SyncManager] = None
        # In order to set unique IDs to workers:
        self.worker_id_queue: Optional[Queue] = None
        # In order to recycle unused but pre-calculated batches from last epoch for training:
        self.multi_worker_cache: Optional[Iterator[DataBatch]] = None

        if self.num_workers > 0:
            # generate unique ids for processes
            self.worker_manager = multiprocessing.Manager()
            self.worker_id_queue = self.worker_manager.Queue()
            for i in range(self.num_workers):
                self.worker_id_queue.put(i)

            # Use multiprocessing.get_context("spawn").Pool to check whether
            # implementation `clean`, i.e no unix forking magic required,
            # Otherwise use recommended defaults
            self.worker_pool = multiprocessing.Pool(
                self.num_workers,
                initializer=_worker_initializer,
                initargs=[
                    self.dataset,
                    self.transformation,
                    self.num_workers,
                    self.worker_id_queue,
                ],
            )

    def __iter__(self) -> Iterator[DataBatch]:
        self.cycle_num += 1
        if self.num_workers == 0:
            generator = _sequential_sample_generator(
                self.dataset, self.transformation, self.is_train, self.cyclic
            )

            def same_process_iter():
                while True:
                    # take the next batch size elements
                    transformed_samples = list(
                        itertools.islice(
                            generator,
                            self.batch_size * self.num_batches_for_shuffling,
                        )
                    )

                    # shuffle data if appropriate and prepare for batching
                    if self.shuffle:
                        random.shuffle(transformed_samples)
                    transformed_samples_iterator = iter(transformed_samples)

                    # batch the samples
                    for i in range(self.num_batches_for_shuffling):
                        batch_samples = list(
                            itertools.islice(
                                transformed_samples_iterator, self.batch_size
                            )
                        )

                        # terminate if no more batches to be dealt with
                        if len(batch_samples) == 0:
                            return

                        # make them into a single batch
                        batch = _batchify_fn(
                            data=batch_samples,
                            multi_processing=False,
                            dtype=self.dtype,
                            single_process_ctx=self.ctx,
                        )

                        yield batch

            return same_process_iter()
        else:
            # to prevent Mypy complaints
            assert isinstance(self.worker_pool, Pool)

            # multi-worker takes care of asynchronously preparing batches
            # only cache multi_worker for cyclic datasets
            if self.multi_worker_cache is None:
                multi_worker = _MultiWorkerIter(
                    worker_pool=self.worker_pool,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_batches_for_shuffling=self.num_batches_for_shuffling,
                    batchify_fn=_batchify_fn,
                    dtype=self.dtype,
                    ctx=self.ctx,
                    is_train=self.is_train,
                    cyclic=self.cyclic,
                    worker_fn=_worker_fn,
                    num_prefetch=self.num_prefetch,
                    dataset_len=self.dataset_len,
                    cycle_num=self.cycle_num,
                    timeout=120,
                )
                if self.cyclic:
                    self.multi_worker_cache = iter(multi_worker)
                return iter(multi_worker)
            else:
                # This way we can recycle the unused pre-fetched batches for the next epoch
                # (cycle num is irrelevant for cyclic datasets, and rest of the arguments stays same between epochs)
                return self.multi_worker_cache

    def __len__(self) -> int:
        return self.dataset_len

    def __del__(self):
        if self.worker_pool:
            # clean up worker pool resource
            assert isinstance(self.worker_pool, multiprocessing.pool.Pool)
            self.worker_pool.close()
            self.worker_pool.join()
