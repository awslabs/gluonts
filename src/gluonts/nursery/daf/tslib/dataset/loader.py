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


from typing import NamedTuple, Callable, Optional, Iterator
import re

from torch import Tensor
from torch._six import container_abcs, int_classes, string_classes
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch as pt

from ..engine.distributed import is_distributed, get_world_size

_default_collate_err_msg_format = (
    "_default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)
np_str_obj_array_pattern = re.compile(r"[SaUO]")


def _default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, pt.Tensor):
        out = None
        if pt.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return pt.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    _default_collate_err_msg_format.format(elem.dtype)
                )
            return _default_collate([pt.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return pt.as_tensor(batch)
    elif isinstance(elem, float):
        return pt.tensor(batch, dtype=pt.float)
    elif isinstance(elem, int_classes):
        return pt.tensor(batch, dtype=pt.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: _default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(_default_collate(samples) for samples in zip(*batch))
        )
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [_default_collate(samples) for samples in transposed]
    elif elem is None:
        return None

    raise TypeError(_default_collate_err_msg_format.format(elem_type))


def copy_to_gpu(data, cuda_device: int, non_blocking: bool):
    if not pt.cuda.is_available():
        raise SystemError("GPU is not available!")
    if isinstance(data, pt.Tensor):
        return data.cuda(cuda_device, non_blocking=non_blocking)
    elif isinstance(data, container_abcs.Mapping):
        return {
            key: copy_to_gpu(val, cuda_device, non_blocking)
            for key, val in data.items()
        }
    elif isinstance(data, container_abcs.Sequence):
        return [copy_to_gpu(t, cuda_device, non_blocking) for t in data]
    elif data is None:
        return None
    else:
        raise ValueError(
            f"expected tensor, sequence or dictionary of tensors, received {type(data).__name__}"
        )


class MetaDataset(NamedTuple):
    """
    Dataset Split Manager. Possess train/valid/test datasets and provide data loaders

    Parameters:
    --------------
    train_set: torch.utils.data.Dataset
        training dataset
    valid_set: torch.utils.data.Dataset or None
        validation dataset if provided
    test_set: torch.utils.data.Dataset
        test dataset
    collate_fn: callable
        function that accepts a batch of inputs and
        returns a stacked version for each data field
    always_validation: bool
        if True, use test data at request of validation loader;
        otherwise return None
    """

    train_set: Dataset
    valid_set: Optional[Dataset]
    test_set: Dataset
    collate_fn: Callable[..., Optional[Tensor]] = _default_collate
    always_validation: bool = False

    def __getattr__(self, key: str):
        return getattr(self.train_set, key)

    @property
    def train_size(self) -> int:
        return len(self.train_set)

    @property
    def valid_size(self) -> Optional[int]:
        if self.valid_set is None:
            raise AttributeError("No validation set")
        return len(self.valid_set)

    @property
    def test_size(self) -> int:
        return len(self.test_set)

    def _data_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        cuda_device: int,
        is_training: bool,
        n_workers: int,
        n_batches: Optional[int],
    ) -> Iterator:
        if is_distributed() and is_training:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            batch_size //= get_world_size()
            shuffle = False
        else:
            sampler = None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=n_workers,
            collate_fn=self.collate_fn,
            pin_memory=cuda_device >= 0,
            drop_last=n_batches is not None,
        )
        if n_batches is None:
            n_batches = len(loader)
        iterator = iter(loader)
        for batch in range(n_batches):
            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                data = next(iterator)
            if cuda_device >= 0:
                data = copy_to_gpu(data, cuda_device, non_blocking=True)
            yield data

    def train_loader(
        self,
        batch_size: int,
        shuffle: bool,
        cuda_device: int,
        n_workers: int = 0,
        n_batches: Optional[int] = None,
    ) -> Iterator:
        return self._data_loader(
            self.train_set,
            batch_size,
            shuffle,
            cuda_device,
            True,
            n_workers,
            n_batches,
        )

    def valid_loader(
        self,
        batch_size: int,
        cuda_device: int,
        n_workers: int = 0,
    ) -> Iterator:
        if self.valid_set is None:
            if self.always_validation:
                dataset = self.test_set
            else:
                raise AttributeError("No validation set")
        else:
            dataset = self.valid_set
        return self._data_loader(
            dataset,
            batch_size,
            False,
            cuda_device,
            False,
            n_workers,
            None,
        )

    def test_loader(
        self,
        batch_size: int,
        cuda_device: int,
        n_workers: int = 0,
    ) -> Iterator:
        return self._data_loader(
            self.test_set,
            batch_size,
            False,
            cuda_device,
            False,
            n_workers,
            None,
        )
