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
import time
from multiprocessing.reduction import ForkingPickler, DupFd, recv_handle
from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import Queue
from typing import Callable, Iterable
import copy

import numpy as np
from gluonts.core.component import DType
from gluonts.dataset.common import Dataset, FileDataset, ListDataset
from gluonts.dataset.util import ReplicaInfo
from gluonts.transform import Transformation
from mxnet.ndarray import NDArray

try:
    import multiprocessing.resource_sharer
except ImportError:
    pass

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
            return mx.nd.array(data, dtype=data.dtype)

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


# NOT SHARED ACROSS PROCESSES !!!
_worker_dataset = None
_worker_dataset_iterator = None
_worker_iterator_reset_num = None
_worker_iterator_exhausted_indicator = None
_worker_tansformation = (
    None  # TODO: maybe unnecessary, added during InferenceDataLoader debug
)


def _worker_initializer(
    dataset: Dataset,
    dataset_len: int,
    num_workers: int,
    transformation: Transformation,
    cyclic: bool,
    worker_id_queue: Queue,
):
    """Initialier for processing pool."""
    # global dataset is per-process based and only available in worker processes
    # this is only necessary to handle MXIndexedRecordIO because otherwise dataset
    # can be passed as argument

    # TODO: remove debug print
    # print(
    #     "num_workers: ",
    #     num_workers,
    #     "current worker: ",
    #     str(multiprocessing.current_process()),
    #     "dataset_id",
    #     dataset,
    #     "batch_size: ",
    #     batch_size,
    #     "is_train: ",
    #     is_train,
    #     "cyclic: ",
    #     cyclic,
    # )

    global _worker_dataset
    global _worker_tansformation
    global _worker_iterator_reset_num

    # replicate dataset
    _worker_dataset = copy.deepcopy(dataset)
    # replicate transformation
    _worker_tansformation = copy.deepcopy(transformation)
    # indicates how often the iterator has been reset
    _worker_iterator_reset_num = 0

    # get unique worker id
    worker_id = int(worker_id_queue.get())
    multiprocessing.current_process().name = f"worker_{worker_id}"

    # propagate worker information to dataset
    if isinstance(_worker_dataset, (FileDataset, ListDataset)):
        start_index = int(
            (worker_id / num_workers) * dataset_len
        )  # calculate offsets for different replicas
        end_index = (
            None
            if cyclic
            else int(((worker_id + 1) / num_workers) * dataset_len)
        )  # loop infinitely if cyclic
        _worker_dataset.set_replica_info(
            ReplicaInfo(
                start_index=start_index,
                end_index=end_index,
                replica_id=worker_id,
            )
        )


def sequential_sample_generator(dataset, transformation, is_train, cyclic):
    while True:
        for sample in transformation(data_it=dataset, is_train=is_train):
            yield sample
        # Dont cycle if not training time
        if not cyclic:
            return


# TODO: for some reason cannot pickle 'future_observed_values' or 'future_target' when using InferenceDataLoader
def _worker_fn(
    batch_size: int,
    batchify_fn: Callable,
    dtype: DType,
    is_train: bool,
    cyclic: bool,
    cycle_num: int,
):
    """Function for processing data in worker process."""
    # pylint: disable=unused-argument
    # it is required that each worker process has to fork a new MXIndexedRecordIO handle
    # preserving dataset as global variable can save tons of overhead and is safe in new process

    global _worker_dataset_iterator
    global _worker_iterator_reset_num
    global _worker_iterator_exhausted_indicator

    # TODO: remove debug print
    # print(
    #     multiprocessing.current_process().name,
    #     "iterator none:",
    #     _worker_dataset_iterator is None,
    # )

    # initialize, or reset the iterator at each cycle
    assert isinstance(_worker_iterator_reset_num, int)
    if (_worker_iterator_reset_num < cycle_num) and (
        cycle_num == 1 or not cyclic
    ):
        # TODO: remove debug print
        # print(
        #     "CYCLE NUM: ",
        #     cycle_num,
        #     "REPLICA ID:",
        #     _worker_dataset.get_replica_info().replica_id,
        #     "worker reset num: ",
        #     _worker_iterator_reset_num,
        #     "cycle_num",
        #     cycle_num,
        # )
        _worker_reset_iterator(is_train, cyclic)

    assert isinstance(
        _worker_dataset_iterator, Iterable
    ), "Dataset not Iterable."
    transformed_data = list(
        itertools.islice(_worker_dataset_iterator, batch_size)
    )

    if transformed_data:
        success = True
        batch = batchify_fn(
            data=transformed_data, dtype=dtype, parallel_processing=True
        )
    else:
        # the second time without being able to provide a batch we want to delay calling them again
        # on fist exhaustion they should not be delayed, since they need to indicate depletion
        if _worker_iterator_exhausted_indicator:
            time.sleep(0.05)
        else:
            _worker_iterator_exhausted_indicator = True
        success = False
        batch = None

    assert isinstance(_worker_dataset, (FileDataset, ListDataset))
    dataset_replica_id = _worker_dataset.get_replica_info().replica_id

    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(
        (success, dataset_replica_id, batch)
    )
    return buf.getvalue()


# initialize or reset iterators
# needed because some iterators are not cyclic
def _worker_reset_iterator(
    is_train: bool, cyclic: bool,
):
    global _worker_dataset
    global _worker_dataset_iterator
    global _worker_tansformation
    global _worker_iterator_reset_num
    global _worker_iterator_exhausted_indicator

    _worker_dataset_iterator = sequential_sample_generator(
        dataset=_worker_dataset,
        transformation=_worker_tansformation,
        is_train=is_train,
        cyclic=cyclic,
    )
    assert isinstance(_worker_iterator_reset_num, int)
    _worker_iterator_reset_num += 1
    # indicates whether the iterator was previously depleted
    _worker_iterator_exhausted_indicator = False


# TODO: test that threads terminate correctly (merged code of mxnet 1.4 and newest
#  which contained more thread termination handling)
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
        cyclic: bool,
        cycle_num: int,
        prefetch: int,
        worker_fn: Callable = _worker_fn,
        dataset_len: int = None,
        timeout: int = 120,
    ):
        self._worker_pool = worker_pool
        self._batchify_fn = batchify_fn
        self._data_buffer: dict = (
            {}
        )  # Its a dictionary with {index: data} structure in our case
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._worker_fn = worker_fn
        self._timeout = timeout

        self.is_train = is_train
        self.dtype = dtype
        self.ctx = ctx
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_len = dataset_len

        # in case of cyclic=False iterators can be exhausted
        self._exhausted_iterators: set = set()

        # pre-fetch
        self.cycle_num = cycle_num
        self.prefetch = prefetch
        for i in range(self.prefetch):
            self._push_next()

    def __len__(self):
        return self.dataset_len

    def _push_next(self):
        """Assign next batch workload to workers."""
        # Optimally one would want to task worker that have none depleted iterators,
        # however, this does not seem to be possible with a worker pool
        async_ret = self._worker_pool.apply_async(
            self._worker_fn,
            (
                self.batch_size,
                self._batchify_fn,
                self.dtype,
                self.is_train,
                self.cyclic,
                self.cycle_num,
            ),
        )
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def __next__(self):
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

                success, dataset_id, batch = pickle.loads(got)

                # If iterator exhausted/empty
                if not success:
                    self._exhausted_iterators.add(dataset_id)
                    if self.num_workers == len(self._exhausted_iterators):
                        # No more batches to be generated
                        return []
                    else:
                        self._push_next()
                else:
                    # either pin to cpu memory, or return with the right context straight away
                    batch = {
                        k: v.as_in_context(self.ctx)
                        if isinstance(
                            v, NDArray
                        )  # context.cpu_pinned(self.pin_device_id)
                        else v
                        for k, v in batch.items()
                    }
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

    def __iter__(self):
        while True:
            next_batch = next(self)
            if len(next_batch) == 0:
                return
            yield next_batch


# TODO: think about how a multiprocessing.Manager() would complement this implementation
class ParallelDataLoader(object):
    """Loads data from a dataset and returns mini-batches of data.
    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transformation
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
    """

    def __init__(
        self,
        dataset: Dataset,
        transformation: Transformation,
        cyclic: bool,
        is_train: bool,
        batch_size: int,
        shuffle: bool = False,
        batchify_fn: Callable = None,
        ctx: mx.Context = None,
        dtype: DType = np.float32,
        prefetch: int = None,
        num_workers: int = 0,
    ):
        self.cyclic = (
            cyclic  # indicates that we want to cycle through the dataset
        )
        self.dtype = dtype
        self.is_train = is_train
        self.transformation = transformation
        self.ctx = ctx
        self.batch_size = batch_size

        self.num_workers = num_workers if num_workers >= 0 else 0
        self.worker_pool = None
        self.prefetch = max(
            0, int(prefetch) if prefetch is not None else 2 * self.num_workers
        )

        self.dataset = dataset
        self.dataset_len = len(list(dataset))

        # indicates the current cycle, needed for resetting iterators at each cycle
        self.cycle_num = 0

        # generate unique ids for processes
        self.worker_id_queue: Queue = Queue()
        for i in range(num_workers):
            self.worker_id_queue.put(i)

        if self.num_workers > 0:
            self.worker_pool = Pool(
                self.num_workers,
                initializer=_worker_initializer,
                initargs=[
                    self.dataset,
                    self.dataset_len,
                    self.num_workers,
                    self.transformation,
                    self.cyclic,
                    self.worker_id_queue,
                ],
            )
        if batchify_fn is None:
            self.batchify_fn = default_batchify_fn
        else:
            self.batchify_fn = batchify_fn

    def __iter__(self):
        self.cycle_num += 1
        if self.num_workers == 0:
            generator = sequential_sample_generator(
                self.dataset, self.transformation, self.is_train, self.cyclic
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
                    batch = self.batchify_fn(
                        data=sample_batch,
                        parallel_processing=False,
                        dtype=self.dtype,
                    )

                    # either pin to cpu memory, or return with the right context straight away
                    batch = {
                        k: v.as_in_context(self.ctx)
                        if isinstance(
                            v, NDArray
                        )  # context.cpu_pinned(self.pin_device_id)
                        else v
                        for k, v in batch.items()
                    }
                    yield batch

            return same_process_iter()
        else:
            # multi-worker takes care of asynchronously preparing batches
            multi_worker = _MultiWorkerIter(
                worker_pool=self.worker_pool,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                batchify_fn=self.batchify_fn,
                dtype=self.dtype,
                ctx=self.ctx,
                is_train=self.is_train,
                cyclic=self.cyclic,
                worker_fn=_worker_fn,
                prefetch=self.prefetch,
                dataset_len=self.dataset_len,
                cycle_num=self.cycle_num,
            )

            return iter(multi_worker)

    def __len__(self):
        return self.dataset_len

    def __del__(self):
        if self.worker_pool:
            # manually terminate due to a bug that pool is not automatically terminated
            # https://bugs.python.org/issue34172
            assert isinstance(self.worker_pool, multiprocessing.pool.Pool)
            self.worker_pool.terminate()
