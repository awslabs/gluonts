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

import pickle
import io
import sys
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
from multiprocessing.pool import ThreadPool
import threading
import numpy as np

try:
    import multiprocessing.resource_sharer
except ImportError:
    pass

from mxnet.gluon.data import sampler as _sampler
from mxnet import nd, context

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
            fd = multiprocessing.reduction.DupFd(fd)
        return rebuild_ndarray, (pid, fd, shape, dtype)


ForkingPickler.register(nd.NDArray, reduce_ndarray)


class ConnectionWrapper(object):
    """Connection wrapper for multiprocessing that supports sending
    NDArray via shared memory."""

    def __init__(self, conn):
        self._conn = conn

    def send(self, obj):
        """Send object"""
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        """Receive object"""
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        """Emmulate conn"""
        attr = self.__dict__.get("_conn", None)
        return getattr(attr, name)


class Queue(multiprocessing.queues.Queue):
    """Wrapper for multiprocessing queue that dumps NDArray with shared memory."""

    def __init__(self, *args, **kwargs):
        if sys.version_info[0] <= 2:
            super(Queue, self).__init__(*args, **kwargs)
        else:
            super(Queue, self).__init__(
                *args, ctx=multiprocessing.get_context(), **kwargs
            )
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    """Wrapper for multiprocessing SimpleQueue that dumps NDArray with shared memory.
       SimpleQueue don't use threading internally.
    """

    def __init__(self, *args, **kwargs):
        if sys.version_info[0] <= 2:
            super(SimpleQueue, self).__init__(*args, **kwargs)
        else:
            super(SimpleQueue, self).__init__(
                *args, ctx=multiprocessing.get_context(), **kwargs
            )
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


def default_batchify_fn(data):
    """Collate data into batch."""
    if isinstance(data[0], nd.NDArray):
        return nd.stack(*data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        return nd.array(data, dtype=data.dtype)


def default_mp_batchify_fn(data):
    """Collate data into batch. Use shared memory for stacking."""
    if isinstance(data[0], nd.NDArray):
        out = nd.empty(
            (len(data),) + data[0].shape,
            dtype=data[0].dtype,
            ctx=context.Context("cpu_shared", 0),
        )
        return nd.stack(*data, out=out)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_mp_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        return nd.array(
            data, dtype=data.dtype, ctx=context.Context("cpu_shared", 0)
        )


def _as_in_context(data, ctx):
    """Move data into new context."""
    if isinstance(data, nd.NDArray):
        return data.as_in_context(ctx)
    elif isinstance(data, (list, tuple)):
        return [_as_in_context(d, ctx) for d in data]
    return data


_worker_dataset = None


def _worker_initializer(dataset):
    """Initialier for processing pool."""
    # global dataset is per-process based and only available in worker processes
    # this is only necessary to handle MXIndexedRecordIO because otherwise dataset
    # can be passed as argument
    global _worker_dataset
    _worker_dataset = dataset


def _worker_fn(samples, batchify_fn, dataset=None):
    """Function for processing data in worker process."""
    # pylint: disable=unused-argument
    # it is required that each worker process has to fork a new MXIndexedRecordIO handle
    # preserving dataset as global variable can save tons of overhead and is safe in new process
    global _worker_dataset
    batch = batchify_fn([_worker_dataset[i] for i in samples])
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(batch)
    return buf.getvalue()


def _thread_worker_fn(samples, batchify_fn, dataset):
    """Threadpool worker function for processing data."""
    return batchify_fn([dataset[i] for i in samples])


###########################################


class _MultiWorkerIter(object):
    """Internal multi-worker iterator for DataLoader."""

    def __init__(
        self,
        worker_pool,
        batchify_fn,
        batch_sampler,
        pin_memory=False,
        pin_device_id=0,
        worker_fn=_worker_fn,
        prefetch=0,
        dataset=None,
    ):
        self._worker_pool = worker_pool
        self._batchify_fn = batchify_fn  # Need to customize
        self._batch_sampler = batch_sampler  # Need to customize
        self._data_buffer = {}  # Its a dictionary with {index: data} structure
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)  # Need to customize
        self._worker_fn = worker_fn
        self._pin_memory = pin_memory
        self._pin_device_id = pin_device_id
        self._dataset = dataset
        # pre-fetch
        for _ in range(prefetch):
            self._push_next()

    def __len__(self):
        return len(self._batch_sampler)

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter, None)  # r is 'samples'
        if r is None:
            return
        async_ret = self._worker_pool.apply_async(
            self._worker_fn,
            (r, self._batchify_fn, self._dataset),  # r is 'samples'
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
            if self._dataset is None:
                batch = pickle.loads(ret.get(self._timeout))
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


class DataLoader(object):
    """Loads data from a dataset and returns mini-batches of data.
    Parameters
    ----------
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batch_size : int
        Size of mini-batch.
    shuffle : bool
        Whether to shuffle the samples.
    sampler : Sampler
        The sampler to use. Either specify sampler or shuffle, not both.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.
        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_batchify_fn`::
            def default_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [default_batchify_fn(i) for i in data]
                else:
                    data = np.asarray(data)
                    return nd.array(data, dtype=data.dtype)
    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory.
    pin_device_id : int, default 0
        The device id to use for allocating pinned memory if pin_memory is ``True``
    prefetch : int, default is `num_workers * 2`
        The number of prefetching batches only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain batches before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more shared_memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_workers` in this case.
        By default it defaults to `num_workers * 2`.
    thread_pool : bool, default False
        If ``True``, use threading pool instead of multiprocessing pool. Using threadpool
        can avoid shared memory usage. If `DataLoader` is more IO bounded or GIL is not a killing
        problem, threadpool version may achieve better performance than multiprocessing.
    """

    def __init__(
        self,
        dataset,
        batch_size=None,
        shuffle=False,
        sampler=None,
        last_batch=None,
        batch_sampler=None,
        batchify_fn=None,
        num_workers=0,
        pin_memory=False,
        pin_device_id=0,
        prefetch=None,
        thread_pool=False,
    ):
        self._dataset = dataset
        self._pin_memory = pin_memory
        self._pin_device_id = pin_device_id
        self._thread_pool = thread_pool

        if batch_sampler is None:
            if batch_size is None:
                raise ValueError(
                    "batch_size must be specified unless "
                    "batch_sampler is specified"
                )
            if sampler is None:
                if shuffle:
                    sampler = _sampler.RandomSampler(len(dataset))
                else:
                    sampler = _sampler.SequentialSampler(len(dataset))
            elif shuffle:
                raise ValueError(
                    "shuffle must not be specified if sampler is specified"
                )

            batch_sampler = _sampler.BatchSampler(
                sampler, batch_size, last_batch if last_batch else "keep"
            )
        elif (
            batch_size is not None
            or shuffle
            or sampler is not None
            or last_batch is not None
        ):
            raise ValueError(
                "batch_size, shuffle, sampler and last_batch must "
                "not be specified if batch_sampler is specified."
            )

        self._batch_sampler = batch_sampler
        self._num_workers = num_workers if num_workers >= 0 else 0
        self._worker_pool = None
        self._prefetch = max(
            0, int(prefetch) if prefetch is not None else 2 * self._num_workers
        )
        if self._num_workers > 0:
            if self._thread_pool:
                self._worker_pool = ThreadPool(self._num_workers)
            else:
                self._worker_pool = multiprocessing.Pool(
                    self._num_workers,
                    initializer=_worker_initializer,
                    initargs=[self._dataset],
                )
        if batchify_fn is None:
            if num_workers > 0:
                self._batchify_fn = default_mp_batchify_fn
            else:
                self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_workers == 0:

            def same_process_iter():
                for batch in self._batch_sampler:
                    ret = self._batchify_fn(
                        [self._dataset[idx] for idx in batch]
                    )
                    if self._pin_memory:
                        ret = _as_in_context(
                            ret, context.cpu_pinned(self._pin_device_id)
                        )
                    yield ret

            return same_process_iter()

        # multi-worker
        return _MultiWorkerIter(
            self._worker_pool,
            self._batchify_fn,
            self._batch_sampler,
            pin_memory=self._pin_memory,
            pin_device_id=self._pin_device_id,
            worker_fn=_thread_worker_fn if self._thread_pool else _worker_fn,
            prefetch=self._prefetch,
            dataset=self._dataset if self._thread_pool else None,
        )

    def __len__(self):
        return len(self._batch_sampler)

    def __del__(self):
        if self._worker_pool:
            # manually terminate due to a bug that pool is not automatically terminated
            # https://bugs.python.org/issue34172
            assert isinstance(self._worker_pool, multiprocessing.pool.Pool)
            self._worker_pool.terminate()
