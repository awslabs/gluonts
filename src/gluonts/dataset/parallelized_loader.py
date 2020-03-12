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
import itertools
import pickle
import io
import sys
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler, DupFd, recv_handle
from multiprocessing.pool import ThreadPool, Pool
from typing import Optional
import copy

import numpy as np
from gluonts.core.component import DType
from gluonts.dataset.common import Dataset, FileDataset, ListDataset
from gluonts.dataset.util import WorkerInfo
from gluonts.transform import Transformation

try:
    import multiprocessing.resource_sharer
except ImportError:
    pass

from mxnet.gluon.data import sampler as _sampler, Sampler
from mxnet import nd, context
import mxnet as mx

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
        if sys.version_info[0] == 2:
            fd = multiprocessing.reduction.rebuild_handle(fd)
        else:
            fd = fd.detach()
        return nd.NDArray(
            nd.ndarray._new_from_shared_mem(pid, fd, shape, dtype)
        )

    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        # keep a local ref before duplicating fd
        data = data.as_in_context(context.Context("cpu_shared", 0))
        pid, fd, shape, dtype = data._to_shared_mem()
        if sys.version_info[0] == 2:
            fd = multiprocessing.reduction.reduce_handle(fd)
        else:
            fd = DupFd(fd)
        return rebuild_ndarray, (pid, fd, shape, dtype)


ForkingPickler.register(nd.NDArray, reduce_ndarray)


# TODO: think about this, especially about possibilities for improvement
# Used when creating a single batch from list of dicts
# depending on whether multiprocessing is turned on, the batches will be
# constructed using different memory allocation techniques
def stack(data, parallel_processing, dtype):
    """Stack a list of data."""
    if isinstance(data[0], np.ndarray):
        data = np.asarray(data)
        if data.dtype.kind == "f":
            data = data.astype(dtype)
        if parallel_processing:
            return mx.nd.array(
                data, dtype=data.dtype, ctx=context.Context("cpu_shared", 0)
            )
        else:
            return mx.nd.array(
                data, dtype=data.dtype
            )  # TODO: Dont use the ctx after all?

    if isinstance(data[0], mx.nd.NDArray):
        if mx:
            out = nd.empty(
                (len(data),) + data[0].shape,
                dtype=data[0].dtype,
                ctx=context.Context("cpu_shared", 0),
            )
            return mx.nd.stack(*data, out=out)
        else:
            return mx.nd.stack(*data)

    # TODO: think about converting int/float lists/tuples to np.NDArray

    if isinstance(data[0], list):
        return list(stack(t, parallel_processing, dtype) for t in zip(*data))

    if isinstance(data[0], tuple):
        return tuple(stack(t, parallel_processing, dtype) for t in zip(*data))

    return data


# Need to define function explicitly, because lambda functions are no pickle'able in some cases
def default_batchify_fn(data, dtype, parallel_processing):
    # reduce the list of dictionaries to a single dictionary, where values
    # referenced by identical key are reduced using the stack function
    return {
        key: stack(
            data=[item[key] for item in data],
            parallel_processing=parallel_processing,
            dtype=dtype,
        )
        for key in data[0].keys()
    }


def _as_in_context(data, ctx):
    """Move data into new context."""
    if isinstance(data, nd.NDArray):
        return data.as_in_context(ctx)
    elif isinstance(data, (list, tuple)):
        return [_as_in_context(d, ctx) for d in data]
    return data


_worker_dataset_list = None


# TODO: make sure they have different offsets? currently they all handle different ids
# TODO: when resampling = False, mechanics is needed not to request same dataset again, if last
#  sample was drawn; put finished (0 vs 1) into shared array for worker id
def _worker_initializer(
    dataset, num_workers, batch_size, transformation, is_train, resample
):
    """Initialier for processing pool."""
    # global dataset is per-process based and only available in worker processes
    # this is only necessary to handle MXIndexedRecordIO because otherwise dataset
    # can be passed as argument

    print(num_workers, batch_size, is_train, resample)

    global _worker_dataset_list

    # replicate dataset for each worker
    temporary_dataset_list = [
        copy.deepcopy(dataset) for i in range(num_workers)
    ]
    # associate each dataset with a worker # TODO: just give each dataset copy an id
    for worker_id, ds in enumerate(temporary_dataset_list):
        print("PRINTING WORKER ID", worker_id)
        if isinstance(ds, (FileDataset, ListDataset)):
            ds.set_worker_info(
                WorkerInfo(
                    worker_id=worker_id,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            )
    # create generators by applying transformation lazily
    _worker_dataset_list = [
        sequential_sample_generator(
            dataset=dataset,
            transformation=transformation,
            is_train=is_train,
            resample=resample,
        )
        for dataset in temporary_dataset_list
    ]


# multiprocessing.current_process()
def _worker_fn(
    dataset_id,
    batchify_fn: callable,
    batch_size,
    transformation: Transformation,
    dtype: DType,
    is_train: bool,
    dataset,
):
    """Function for processing data in worker process."""
    # pylint: disable=unused-argument
    # it is required that each worker process has to fork a new MXIndexedRecordIO handle
    # preserving dataset as global variable can save tons of overhead and is safe in new process

    global _worker_dataset_list

    transformed_data = list(
        itertools.islice(_worker_dataset_list[dataset_id], batch_size)
    )

    batch = batchify_fn(
        data=transformed_data, dtype=dtype, parallel_processing=True
    )
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(batch)
    return buf.getvalue()


# TODO: delete this
def _thread_worker_fn(
    batchify_fn: callable,
    transformation: Transformation,
    dtype: DType,
    is_train: bool,
    dataset,
):
    """Threadpool worker function for processing data."""
    # data = [dataset[i] for i in samples]
    # transformed_data = list(transformation(data_it=data, is_train=is_train))

    if isinstance(dataset, (FileDataset, ListDataset)):
        transformed_data = list(
            itertools.islice(dataset, dataset.worker_info.batch_size)
        )

        batch = batchify_fn(
            data=transformed_data, dtype=dtype, parallel_processing=True
        )
        return batch


def sequential_sample_generator(dataset, transformation, is_train, resample):
    while True:
        for sample in transformation(data_it=dataset, is_train=is_train):
            yield sample
        # Dont cycle if not training time
        if not resample:
            return


# TODO: test that threads terminate correctly (merged code of mxnet 1.4 and newest
#  which contained more thread termination handling)
class _MultiWorkerIter(object):
    """Internal multi-worker iterator for DataLoader."""

    def __init__(
        self,
        worker_pool: Pool,
        batchify_fn: callable,
        transform: Transformation,  # yield Iterator of transformed Dataset
        dtype: DType,
        is_train: bool,
        pin_memory: Optional[bool] = False,
        pin_device_id: Optional[bool] = 0,
        worker_fn: Optional[callable] = _worker_fn,
        dataset: Optional[Dataset] = None,
        prefetch: Optional[int] = 0,
        timeout: Optional[int] = 120,
        num_workers: int = None,
        batch_size: int = None,
        resample: bool = None,
        thread_pool: bool = None,
    ):
        self._worker_pool = worker_pool
        self.thread_pool = thread_pool
        self._batchify_fn = batchify_fn  # Need to customize
        self.transform = transform
        self.dtype = dtype
        self.is_train = is_train
        self.resample = resample
        self._data_buffer = (
            {}
        )  # Its a dictionary with {index: data} structure in our case
        self._rcvd_idx = 0
        self._sent_idx = 0
        # self._iter = iter(self._batch_sampler)
        self._worker_fn = worker_fn
        self._pin_memory = pin_memory
        self._pin_device_id = pin_device_id
        self._dataset = dataset
        self._timeout = timeout

        self.num_workers = num_workers
        self.batch_size = batch_size

        # cycle dataset ids
        self._iter = itertools.cycle(range(num_workers))

        # pre-fetch
        for _ in range(prefetch):
            self._push_next()

    def __len__(self):
        return len(self._batch_sampler)

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter)  # next(self._iter, None)  # r is now dataset id
        if r is None:
            return
        async_ret = self._worker_pool.apply_async(
            self._worker_fn,
            (
                r,
                self._batchify_fn,
                self.resample,
                self.transform,
                self.dtype,
                self.is_train,
                self._dataset,
            ),  # r is now dataset id
        )
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def __next__(self):
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
        try:
            if (
                self._dataset is None
            ):  # not self.thread_pool:  # self._dataset is None:
                got = ret.get(self._timeout)
                batch = pickle.loads(got)
            else:
                batch = ret.get(self._timeout)
            if self._pin_memory:
                batch = _as_in_context(
                    batch, context.cpu_pinned(self._pin_device_id)
                )
            self._rcvd_idx += 1
            return batch
        except multiprocessing.context.TimeoutError:
            msg = """Worker timed out after {} seconds. This might be caused by \n
            - Slow transform. Please increase timeout to allow slower data loading in each worker.
            """.format(
                self._timeout
            )
            if not isinstance(
                self._worker_pool, multiprocessing.pool.ThreadPool
            ):
                msg += """- Insufficient shared_memory if `timeout` is large enough.
            Please consider reduce `num_workers` or increase shared_memory in system.
            """
            print(msg)
            raise
        except Exception:
            self._worker_pool.terminate()
            raise

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class ParallelDataLoader(object):
    """Loads data from a dataset and returns mini-batches of data.
    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transform
        A transformation to apply to each entry in the dataset.
    batch_size
        Size of mini-batch.
    ctx
        MXNet context to use to store data.
    dtype
        Floating point type to use.

    shuffle
        Whether to shuffle the samples.
    sampler
        The sampler to use. Either specify sampler or shuffle, not both.
    last_batch
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.
        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    batch_sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn
        Callback function to allow users to specify how to merge samples
        into a batch.
    num_workers
        The number of multiprocessing workers to use for data preprocessing.
    pin_memory
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory.
    pin_device_id
        The device id to use for allocating pinned memory if pin_memory is ``True``
    prefetch
        The number of prefetching batches only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain batches before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more shared_memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_workers` in this case.
        By default it defaults to `num_workers * 2`.
    thread_pool
        If ``True``, use threading pool instead of multiprocessing pool. Using threadpool
        can avoid shared memory usage. If `DataLoader` is more IO bounded or GIL is not a killing
        problem, threadpool version may achieve better performance than multiprocessing.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        is_train: bool,
        ctx: mx.Context,  # TODO: check whether really necessary
        dtype: DType = np.float32,
        batch_size: int = None,
        shuffle: bool = False,
        batchify_fn: Optional[callable] = None,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
        pin_device_id: Optional[int] = 0,
        prefetch: Optional[int] = None,
        thread_pool: Optional[bool] = False,
        resample: bool = False,
    ):
        self.resample = resample
        self.ctx = ctx  # TODO: currently not in use
        self.dtype = dtype
        self.is_train = is_train
        self.transform = transform
        self.batch_size = batch_size

        self.pin_memory = pin_memory
        self.pin_device_id = pin_device_id
        self.thread_pool = thread_pool

        self.num_workers = num_workers if num_workers >= 0 else 0
        self.worker_pool = None
        self.prefetch = max(
            0, int(prefetch) if prefetch is not None else 2 * self.num_workers
        )

        self.dataset = dataset

        if self.num_workers > 0:
            if self.thread_pool:
                self.worker_pool = ThreadPool(self.num_workers)
            else:
                self.worker_pool = Pool(
                    self.num_workers,
                    initializer=_worker_initializer,
                    initargs=[
                        self.dataset,
                        self.num_workers,
                        self.batch_size,
                        self.transform,
                        self.is_train,
                        self.resample,
                    ],
                )
        if batchify_fn is None:
            self.batchify_fn = default_batchify_fn
        else:
            self.batchify_fn = batchify_fn

    def __iter__(self):
        if self.num_workers == 0:
            generator = sequential_sample_generator(
                self.dataset, self.transform, self.is_train, self.resample
            )

            def same_process_iter():
                while True:
                    # take the next batch size elements
                    sample_batch = list(
                        itertools.islice(generator, self.batch_size)
                    )

                    # terminate if no more batches to be dealt with
                    if len(sample_batch) == 0:
                        return

                    # make them into a single batch
                    ret = self.batchify_fn(
                        data=sample_batch,
                        parallel_processing=False,
                        dtype=self.dtype,
                    )

                    # pin them into memory for faster copying into GPU memory
                    if self.pin_memory:
                        ret = _as_in_context(
                            ret, context.cpu_pinned(self.pin_device_id)
                        )
                    yield ret

            return same_process_iter()

        # multi-worker
        return _MultiWorkerIter(
            worker_pool=self.worker_pool,
            num_workers=self.num_workers,
            thread_pool=self.thread_pool,
            batchify_fn=self.batchify_fn,
            transform=self.transform,
            dtype=self.dtype,
            is_train=self.is_train,
            resample=self.resample,
            pin_memory=self.pin_memory,
            pin_device_id=self.pin_device_id,
            worker_fn=_thread_worker_fn if self.thread_pool else _worker_fn,
            prefetch=self.prefetch,
            dataset=self.dataset
            if self.thread_pool
            else None,  # TODO check validity of this
        )

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        if self.worker_pool:
            # manually terminate due to a bug that pool is not automatically terminated
            # https://bugs.python.org/issue34172
            assert isinstance(self.worker_pool, multiprocessing.pool.Pool)
            self.worker_pool.terminate()
