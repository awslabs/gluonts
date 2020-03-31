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
import logging
from typing import Any, Dict, Iterable, Iterator, Optional
from multiprocessing import cpu_count

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import DataEntry, Dataset, FileDataset, ListDataset
from gluonts.transform import Transformation

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
    cyclic
        Indicates whether the dataset is traversed potentially multiple times.

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
        cyclic: bool = False,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs
    ) -> None:
        # conversion from iterator to list
        self.batch_size = batch_size
        self.ctx = ctx
        self.dtype = dtype
        self.is_train = is_train
        self.transform = transform
        self.cyclic = cyclic

        self.parallel_data_loader = ParallelDataLoader(
            dataset=dataset,
            transformation=self.transform,
            cyclic=self.cyclic,
            is_train=self.is_train,
            batch_size=self.batch_size,
            ctx=ctx,
            dtype=self.dtype,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            **kwargs,
        )

    def __iter__(self) -> Iterator[DataBatch]:
        # Will take all batches, so that all data is sampled exactly once
        yield from self.parallel_data_loader


class TrainDataLoader(DataLoader):
    """
    An Iterable type for iterating and transforming a dataset, in batches of a
    prescribed size, until a given number of batches is reached.

    The transformation are applied with in training mode, i.e. with the flag `is_train = True`.

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
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        dtype: DType = np.float32,
        shuffle_for_training: bool = True,
        num_batches_for_shuffling: int = 10,  # TODO: this does not work currently
        **kwargs
    ) -> None:
        assert dataset, "empty dataset"

        super().__init__(
            dataset=dataset,
            transform=transform,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
            is_train=True,
            shuffle=shuffle_for_training,
            cyclic=True,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            **kwargs,
        )

        self.num_batches_per_epoch = num_batches_per_epoch
        self.shuffle_for_training = shuffle_for_training
        # TODO: implement shuffling
        self.num_batches_for_shuffling = num_batches_for_shuffling

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[DataBatch]:
        # take num_batches of batches for one epoch
        return itertools.islice(
            self.parallel_data_loader, self.num_batches_per_epoch
        )


class ValidationDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
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
            cyclic=False,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
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
        # Currently the is a bug with multi processing here,
        # see: _worker_fn in parallelized_loader.py for explanation
        num_workers: Optional[int] = 0,
        num_prefetch: Optional[int] = None,
        dtype: DType = np.float32,
        **kwargs
    ) -> None:
        if num_workers != 0:
            num_workers = 0
            logging.warning(
                "You have set `num_workers` for InferenceDataLoader to a non zero value, "
                "however, currently multiprocessing is not supported for the InferenceDataLoader."
            )

        super().__init__(
            dataset=dataset,
            transform=transform,
            is_train=False,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
            cyclic=False,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            **kwargs,
        )
