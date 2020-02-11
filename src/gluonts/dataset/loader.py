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
import random
from collections import defaultdict
from functools import partial
from typing import Any, Collection, Dict, Iterable, Iterator, List, Optional

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.transform import Transformation

DataBatch = Dict[str, Any]


def batcher(iterable, batch_size):
    while True:
        items = list(itertools.islice(iterable, batch_size))
        if not items:
            break
        yield items


class BatchStacker:
    def __init__(self, batch_size: int, stream, stack_fn) -> None:
        self.batch_size = batch_size
        self.stream = stream
        self.stack = stack_fn

    def __iter__(self):
        for items in batcher(self.stream, self.batch_size):
            yield {
                key: self.stack([item[key] for item in items])
                for key in items[0]
            }


class Shuffler(Iterable[DataBatch]):
    def __init__(self, batch_size, stream):
        self.batch_size = batch_size
        self.stream = stream

    def __iter__(self) -> Iterator[DataBatch]:
        for batch in batcher(self.stream, self.batch_size):
            random.shuffle(batch)
            yield from batch


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
    dtype
        Floating point type to use.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        is_train: bool,
        batch_size: int,
        ctx: mx.Context,
        dtype: DType = np.float32,
    ) -> None:
        self.batch_size = batch_size
        self.ctx = ctx
        self.dtype = dtype
        self.stream: Iterable = transform(dataset, is_train=is_train)

    @property
    def batches(self):
        return BatchStacker(
            self.batch_size, stream=self.stream, stack_fn=self.stack
        )

    def stack(self, xs):
        if isinstance(xs[0], np.ndarray):
            data = np.asarray(xs)
            if data.dtype.kind == "f":
                data = data.astype(self.dtype)
            return mx.nd.array(data, dtype=data.dtype, ctx=self.ctx)

        if isinstance(xs[0], mx.nd.NDArray):
            return mx.nd.stack(*xs)

        if isinstance(xs[0], list):
            return list(self.stack(t) for t in zip(*xs))

        if isinstance(xs[0], tuple):
            return tuple(self.stack(t) for t in zip(*xs))

        return xs

    def __iter__(self) -> Iterator[DataBatch]:
        return iter(self.batches)


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
    dtype
        Floating point type to use.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        num_batches_per_epoch: int,
        dtype: DType = np.float32,
        shuffle_for_training: bool = True,
        num_batches_for_shuffling: int = 10,
    ) -> None:
        assert dataset, "empty dataset"

        super().__init__(
            dataset=itertools.cycle(dataset),
            transform=transform,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
            is_train=True,
        )

        self.num_batches_per_epoch = num_batches_per_epoch

        if shuffle_for_training:
            self.stream = Shuffler(num_batches_for_shuffling, self.stream)

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[DataBatch]:
        return itertools.islice(self.batches, self.num_batches_per_epoch)


class ValidationDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(
            dataset,
            transform=transform,
            is_train=True,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
        )


class InferenceDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(
            dataset,
            transform=transform,
            is_train=False,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
        )
