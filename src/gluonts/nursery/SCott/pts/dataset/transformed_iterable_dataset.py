import itertools
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import torch

from pts.transform.transform import Transformation
from .common import DataEntry, Dataset
from .loader import BatchBuffer


class TransformedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset: Dataset, is_train: bool, transform: Transformation
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        self._cur_iter: Optional[Iterator] = None

    def _iterate_forever(
        self, collection: Iterable[DataEntry]
    ) -> Iterator[DataEntry]:
        # iterate forever over the collection, the collection must be non empty
        while True:
            try:
                first = next(iter(collection))
            except StopIteration:
                raise Exception("empty dataset")
            else:
                for x in itertools.chain([first], collection):
                    yield x

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        if self._cur_iter is None:
            self._cur_iter = self.transform(
                self._iterate_forever(self.dataset), is_train=self.is_train
            )
        assert self._cur_iter is not None
        while True:
            data_entry = next(self._cur_iter)
            yield {
                k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
                for k, v in data_entry.items()
                if isinstance(v, np.ndarray) == True
            }

    def __len__(self) -> int:
        return sum(1 for _ in self.dataset)


class TransformedGroupedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        list_of_dataset,
        is_train: bool,
        transform: Transformation,
        batch_size,
    ) -> None:
        super().__init__()
        self.list_of_dataset = list_of_dataset
        self.transform = transform
        self.is_train = is_train
        self.num_groups = len(list_of_dataset)
        self._cur_iters = [None for i in range(self.num_groups)]
        self.index = -1
        self.t = -1
        self.batch_size = batch_size

    def _iterate_forever(
        self, collection: Iterable[DataEntry]
    ) -> Iterator[DataEntry]:
        # iterate forever over the collection, the collection must be non empty
        while True:
            try:
                first = next(iter(collection))
            except StopIteration:
                raise Exception("empty dataset")
            else:
                for x in itertools.chain([first], collection):
                    yield x

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        for iter_id in range(self.num_groups):
            if self._cur_iters[iter_id] is None:
                self._cur_iters[iter_id] = self.transform(
                    self._iterate_forever(self.list_of_dataset[iter_id]),
                    is_train=self.is_train,
                )
            assert self._cur_iters[iter_id] is not None
        while True:
            self.index = (self.index + 1) % self.num_groups
            data_entry = next(self._cur_iters[self.index])
            yield {
                k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
                for k, v in data_entry.items()
                if isinstance(v, np.ndarray) == True
            }


class FullGroupBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        list_of_dataset: Dataset,
        is_train: bool,
        transform: Transformation,
    ) -> None:
        super().__init__()
        self.list_of_dataset = list_of_dataset
        self.transform = transform
        self.is_train = is_train
        self.num_groups = len(list_of_dataset)
        self._cur_iters = self.transform(
            iter(self.list_of_dataset), is_train=True
        )

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        while True:
            try:
                data_entry = next(self._cur_iters)
                yield {
                    k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
                    for k, v in data_entry.items()
                    if isinstance(v, np.ndarray) == True
                }
            except:
                break


class FullBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset: Dataset, is_train: bool, transform: Transformation
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        for data_entry in self.transform(iter(self.dataset), is_train=True):
            yield {
                k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
                for k, v in data_entry.items()
                if isinstance(v, np.ndarray) == True
            }

    def __len__(self) -> int:
        return sum(1 for _ in self.dataset)
