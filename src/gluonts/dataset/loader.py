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

import logging
import io
import pickle
from tempfile import TemporaryFile
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from pydantic import BaseModel

from gluonts.dataset import DataBatch, Dataset
from gluonts.env import env
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled, batcher
from gluonts.transform import AdhocTransform, Identity, Transformation

logger = logging.getLogger(__name__)


DataLoader = Iterable[DataBatch]


@dataclass
class Cache:
    dataset: Dataset
    _cached: bool = False
    batch_size: int = 1024

    def _iter_write(self):

        batch = []
        for element in self.dataset:
            yield element

            batch.append(element)
            if len(batch) == self.batch_size:
                self.write(batch)

        if batch:
            self.write(batch)

        self._cached = True

    def _iter_cached(self):
        raise NotImplementedError

    def __iter__(self):
        if not self._cached:
            yield from self._iter_write()
        else:
            yield from self._iter_cached()

    def __len__(self):
        return len(self.dataset)


@dataclass
class PickleCache(Cache):
    file: io.IOBase = field(init=False)

    def __post_init__(self):
        self.file = TemporaryFile()

    def write(self, batch):
        pickle.dump(batch, self.file)

    def _iter_cached(self):
        self.file.seek(0)

        while True:
            try:
                yield from pickle.load(self.file)
            except EOFError:
                return


# TODO: the following are for backward compatibility
# and could eventually be removed


class Batch(Transformation, BaseModel):
    batch_size: int

    def __call__(self, data, is_train):
        yield from batcher(data, self.batch_size)


@env._inject(cache="cache_loader")
def TrainDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
    num_batches_per_epoch: Optional[int] = None,
    shuffle_buffer_length: Optional[int] = None,
    cache: bool = False,
):
    """
    Construct an iterator of batches for training purposes.

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
    shuffle_buffer_length
        Size of the buffer used for shuffling. Default: None, in which case no
        shuffling occurs.

    Returns
    -------
    Iterator[DataBatch]
        An iterator of batches.
    """

    if cache:
        dataset = PickleCache(dataset)

    dataset = Cyclic(dataset)

    if shuffle_buffer_length:
        dataset = PseudoShuffled(dataset, shuffle_buffer_length)

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    transformed_dataset = transform.apply(dataset, is_train=True)

    batches = iter(transformed_dataset)
    return IterableSlice(batches, num_batches_per_epoch)


@env._inject(cache="cache_loader")
def ValidationDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
    cache: bool = False,
):
    """
    Construct an iterator of batches for validation purposes.

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

    if cache:
        dataset = PickleCache(dataset)

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    return transform.apply(dataset, is_train=True)


def InferenceDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
):
    """
    Construct an iterator of batches for inference purposes.

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
    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    return transform.apply(dataset, is_train=False)
