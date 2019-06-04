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
import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional  # noqa: F401

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.transform import Transformation

DataBatch = Dict[str, Any]


class BatchBuffer:
    def __init__(
        self, batch_size: int, ctx: mx.Context, float_type: DType = np.float32
    ) -> None:
        self._buffers: Dict[Any, List[Any]] = defaultdict(list)
        self.batch_size = batch_size
        self._size = 0
        self.ctx = ctx
        self.float_type = float_type

    def add(self, d: Dict[str, List[np.ndarray]]):
        if self._buffers:
            assert self._buffers.keys() == d.keys()
        for k, v in d.items():
            self._buffers[k].append(v)
        self._size += 1

    def __len__(self):
        return self._size

    def next_batch(self) -> DataBatch:
        assert self._size > 0
        n = min(self._size, self.batch_size)
        batch = {k: self.stack(v[:n]) for k, v in self._buffers.items()}
        for key in self._buffers.keys():
            self._buffers[key] = self._buffers[key][n:]
        self._size -= n
        return batch

    def stack(self, xs):
        if isinstance(xs[0], np.ndarray):
            data = np.asarray(xs)
            if data.dtype.kind == 'f':
                data = data.astype(self.float_type)
            return mx.nd.array(data, dtype=data.dtype, ctx=self.ctx)
        elif isinstance(xs[0], mx.nd.NDArray):
            return mx.nd.stack(*xs)
        else:
            return xs  # stack all other types as list

    def shuffle(self):
        perm = np.random.permutation(self._size)
        for key in self._buffers.keys():
            li = self._buffers[key]
            self._buffers[key] = [li[i] for i in perm]


class DataLoader(Iterable[DataEntry]):
    """
    An abstract Iterable type for iterating and transforming a dataset,
    in batches of a prescribed size.

    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transform
        A transformation to apply to each entry in the dataset.
    batch_size
        The size of the batches to emit.
    ctx
        MXNet context to use to store data.
    float_type
        Floating point type to use.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        float_type: DType = np.float32,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.batch_size = batch_size
        self.ctx = ctx
        self.float_type = float_type


class TrainDataLoader(DataLoader):
    """
    An Iterable type for iterating and transforming a dataset, in batches of a
    prescribed size, until a given number of batches is reached.

    The transformation are applied with in training mode, i.e. with the flag
    `is_train = True`.

    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transform
        A transformation to apply to each entry in the dataset.
    batch_size
        The size of the batches to emit.
    ctx
        MXNet context to use to store data.
    num_batches_per_epoch
        Number of batches to return in one complete iteration over this object.
    float_type
        Floating point type to use.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        num_batches_per_epoch: int,
        float_type: DType = np.float32,
        shuffle_for_training: bool = True,
        num_batches_for_shuffling: int = 10,
    ) -> None:
        super().__init__(dataset, transform, batch_size, ctx, float_type)
        self.num_batches_per_epoch = num_batches_per_epoch
        self.shuffle_for_training = shuffle_for_training
        self._num_buffered_batches = (
            num_batches_for_shuffling if shuffle_for_training else 1
        )
        self._cur_iter: Optional[Iterator] = None
        self._buffer = BatchBuffer(self.batch_size, ctx, float_type)

    def _emit_batches_while_buffer_larger_than(
        self, thresh
    ) -> Iterator[DataBatch]:
        if self.shuffle_for_training:
            self._buffer.shuffle()
        while len(self._buffer) > thresh:
            yield self._buffer.next_batch()

    def _iterate_forever(
        self, collection: Iterable[DataEntry]
    ) -> Iterator[DataEntry]:
        # iterate forever over the collection, the collection must be non empty
        while True:
            try:
                first = next(iter(collection))
            except StopIteration:
                raise Exception('empty dataset')
            else:
                for x in itertools.chain([first], collection):
                    yield x

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[DataBatch]:
        batch_count = 0
        if self._cur_iter is None:
            self._cur_iter = self.transform(
                self._iterate_forever(self.dataset), is_train=True
            )
        assert self._cur_iter is not None
        while True:
            data_entry = next(self._cur_iter)
            self._buffer.add(data_entry)
            if (
                len(self._buffer)
                >= self._num_buffered_batches * self.batch_size
            ):
                for batch in self._emit_batches_while_buffer_larger_than(
                    self.batch_size - 1
                ):
                    yield batch
                    batch_count += 1
                    if batch_count >= self.num_batches_per_epoch:
                        return


class InferenceDataLoader(DataLoader):
    """
    An Iterable type for iterating and transforming a dataset just once, in
    batches of a prescribed size.

    The transformation are applied with in inference mode, i.e. with the flag
    `is_train = False`.

    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transform
        A transformation to apply to each entry in the dataset.
    batch_size
        The size of the batches to emit.
    ctx
        MXNet context to use to store data.
    float_type
        Floating point type to use.
    """

    def __iter__(self) -> Iterator[DataBatch]:
        buffer = BatchBuffer(self.batch_size, self.ctx, self.float_type)
        for data_entry in self.transform(iter(self.dataset), is_train=False):
            buffer.add(data_entry)
            if len(buffer) >= self.batch_size:
                yield buffer.next_batch()
        if len(buffer) > 0:
            yield buffer.next_batch()
