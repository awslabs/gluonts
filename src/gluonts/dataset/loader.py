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
import functools
import itertools
from typing import Any, Dict, Iterable, Iterator

# Third-party imports
import mxnet as mx
import numpy as np
from multiprocessing import cpu_count

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.transform import Transformation

from .util import take, batcher, dct_reduce, shuffler

DataBatch = Dict[str, Any]

from gluonts.dataset.parallelized_loader import ParallelDataLoader


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
    num_workers
        Number of workers.
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
        num_workers: int = 0,  # cpu_count(),  # TODO: think about this, non default
        pin_memory: bool = True,  # TODO: think about this, non default
        **kwargs
    ) -> None:
        self.batch_size = batch_size
        self.ctx = ctx
        self.dtype = dtype
        self.is_train = is_train
        self.transform = transform

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.last_batch = "rollover" if is_train else "keep"

        self.parallel_data_loader = ParallelDataLoader(
            dataset=dataset,
            transform=self.transform,
            is_train=self.is_train,
            batch_size=self.batch_size,
            ctx=self.ctx,
            dtype=self.dtype,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            last_batch=self.last_batch,
            **kwargs,
        )

    def __iter__(self) -> Iterator[DataBatch]:
        # Will take all batches, so that all data is sampled exactly once if is_train is False
        return take(self.parallel_data_loader, len(self.parallel_data_loader))


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
        Number of batches to return in one complete iteration over this object.  # TODO: this is not what its used for
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
        **kwargs
    ) -> None:
        assert dataset, "empty dataset"

        super().__init__(
            dataset=dataset,  # itertools.cycle(dataset) # GET infinite number of samples
            transform=transform,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
            is_train=True,
            shuffle=shuffle_for_training,
            **kwargs,
        )

        self.num_batches_per_epoch = num_batches_per_epoch
        self.shuffle_for_training = shuffle_for_training
        # self.num_batches_for_shuffling = num_batches_for_shuffling # I dont think we need this anymore

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[DataBatch]:
        # this takes num_batches of batches for one epoch
        # sampling with replacement is handled by the parallel_data_loader
        return take(self.parallel_data_loader, self.num_batches_per_epoch)


class ValidationDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        dtype: DType = np.float32,
        **kwargs
    ) -> None:
        super().__init__(
            dataset=dataset,
            transform=transform,
            is_train=True,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
            **kwargs,
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
        **kwargs
    ) -> None:
        super().__init__(
            dataset=dataset,
            transform=transform,
            is_train=False,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
            **kwargs,
        )
