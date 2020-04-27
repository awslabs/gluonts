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
import multiprocessing as mp

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import DataEntry, Dataset, DataBatch
from gluonts.dataset.parallelized_loader import ParallelDataLoader
from gluonts.transform import Transformation


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
    cyclic
        Indicates whether the dataset is traversed potentially multiple times.

    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        cyclic: bool,
        is_train: bool,
        batch_size: int,
        ctx: mx.Context,
        dtype: DType = np.float32,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.batch_size = batch_size
        self.ctx = ctx
        self.dtype = dtype
        self.is_train = is_train
        self.transform = transform
        self.cyclic = cyclic
        self.logger = logging.getLogger(__name__)
        if num_workers is not None and num_workers > mp.cpu_count():
            self.logger.warning(
                f"num_workers is set to {num_workers}, but there are only {mp.cpu_count()} cpus "
                f"please reduce the number of workers"
            )
        self.num_workers = num_workers
        self.num_prefetch = num_prefetch

        self.parallel_data_loader = ParallelDataLoader(
            dataset=dataset,
            transformation=self.transform,
            cyclic=self.cyclic,
            is_train=self.is_train,
            batch_size=self.batch_size,
            ctx=self.ctx,
            dtype=self.dtype,
            num_workers=self.num_workers,
            num_prefetch=self.num_prefetch,
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
    dtype
        Floating point type to use. Default is np.float32.
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
        **kwargs,
    ) -> None:
        assert dataset, "empty dataset"

        super().__init__(
            dataset=dataset,
            transform=transform,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
            is_train=True,
            cyclic=True,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            **kwargs,
        )

        self.num_batches_per_epoch = num_batches_per_epoch
        self._it = iter(self.parallel_data_loader)

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[DataBatch]:
        i = 0
        while True:
            for batch in self._it:
                yield batch
                i += 1
                if i == self.num_batches_per_epoch:
                    return
            self._it = iter(self.parallel_data_loader)


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
        **kwargs,
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
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        dtype: DType = np.float32,
        **kwargs,
    ) -> None:
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
