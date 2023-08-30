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


from typing import Optional, Tuple, List, Union
from itertools import product
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from torch import Tensor
import torch as pt
import pandas as pd
import numpy as np

from tslib.dataset import (
    TimeSeries,
    TimeSeriesCorpus,
    WindowsDataset,
    MetaDataset,
)
from data import (
    SeasonalReader,
    FourierSignalReader,
    FusedSeasonalSignalReader,
    RandomSeasonalSignalReader,
    TrendFourierSignalReader,
    GluonTSJsonReader,
)


def _tensor_collator(batch: List[Tensor]):
    elem = batch[0]
    lengths = [t.size(0) for t in batch]
    max_length = max(lengths)
    pad_size = [max_length - l for l in lengths]

    storage = elem.storage()._new_shared(
        len(batch) * max_length * elem[0].numel()
    )
    out = elem.new(storage)
    if max(pad_size) == 0:
        tensor = pt.stack(batch, dim=0, out=out)
        lengths = None
    else:
        # shorter sequences are padded at left
        tensor = pt.stack(
            [
                pt.cat(
                    [sample.new_zeros(pad, *sample.shape[1:]), sample], dim=0
                )
                for pad, sample in zip(pad_size, batch)
            ],
            dim=0,
            out=out,
        )
        lengths = tensor.new_tensor(lengths, dtype=pt.long)
    return tensor, lengths


def _variable_length_collator(batch: List[Tuple]):
    data, feats, nan_mask = zip(*batch)
    data, length = _tensor_collator(data)
    if feats[0] is not None:
        feats, _ = _tensor_collator(feats)
    else:
        feats = None
    if nan_mask[0] is not None:
        nan_mask, _ = _tensor_collator(nan_mask)
    else:
        nan_mask = None
    return data, feats, nan_mask, length


def _pair_variable_length_collator(batch: List[Tuple]):
    (
        src_data,
        tgt_data,
        src_feats,
        tgt_feats,
        src_nan_mask,
        tgt_nan_mask,
    ) = zip(*batch)
    if src_data[0] is None:
        src_input = (None, None, None, None)
    else:
        src_input = _variable_length_collator(
            list(zip(src_data, src_feats, src_nan_mask))
        )
    if tgt_data[0] is None:
        tgt_input = (None, None, None, None)
    else:
        tgt_input = _variable_length_collator(
            list(zip(tgt_data, tgt_feats, tgt_nan_mask))
        )
    return sum(zip(src_input, tgt_input), ())


class DomAdaptDataset(TorchDataset):
    def __init__(
        self,
        source_dataset: Optional[WindowsDataset],
        target_dataset: Optional[WindowsDataset],
    ) -> None:
        super(DomAdaptDataset, self).__init__()
        if (source_dataset is None) and (target_dataset is None):
            raise ValueError("Source and target cannot be both None.")
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self._source_size = (
            1 if self.source_dataset is None else len(self.source_dataset)
        )
        self._target_size = (
            1 if self.target_dataset is None else len(self.target_dataset)
        )

    def __len__(self) -> int:
        return self._target_size

    def __getitem__(self, index: int) -> Tuple:
        src_index = np.random.choice(self._source_size)
        tgt_index = index
        src = (
            None
            if self.source_dataset is None
            else self.source_dataset[src_index]
        )
        tgt = (
            None
            if self.target_dataset is None
            else self.target_dataset[tgt_index]
        )
        if src is None:
            src = tuple([None] * len(tgt))
        if tgt is None:
            tgt = tuple([None] * len(src))
        return sum(zip(src, tgt), ())

    @classmethod
    def from_domains(
        cls,
        source_dataset: Optional[MetaDataset],
        target_dataset: Optional[MetaDataset],
    ) -> MetaDataset:
        def _get_subset(
            dataset: Optional[MetaDataset], name: str
        ) -> Optional[WindowsDataset]:
            if dataset is None:
                return None
            return getattr(dataset, name)

        kwargs = {}
        kwargs["train_set"] = cls(
            _get_subset(source_dataset, "train_set"),
            _get_subset(target_dataset, "train_set"),
        )
        kwargs["test_set"] = cls(
            _get_subset(source_dataset, "test_set"),
            _get_subset(target_dataset, "test_set"),
        )
        try:
            kwargs["valid_set"] = cls(
                _get_subset(source_dataset, "valid_set"),
                _get_subset(target_dataset, "valid_set"),
            )
        except ValueError:
            kwargs["valid_set"] = None

        if (source_dataset is not None) and (
            source_dataset.collate_fn is _variable_length_collator
        ):
            kwargs.setdefault("collate_fn", _pair_variable_length_collator)
        if (target_dataset is not None) and (
            target_dataset.collate_fn is _variable_length_collator
        ):
            kwargs.setdefault("collate_fn", _pair_variable_length_collator)

        return MetaDataset(**kwargs)


class SeasonalDataset(WindowsDataset, SeasonalReader):
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        ts = super(SeasonalDataset, self).__getitem__(index)
        data = pt.tensor(ts.target, dtype=pt.float)
        return (data,)

    @property
    def d_data(self) -> int:
        return 1

    @property
    def d_feats(self) -> int:
        return 0

    @classmethod
    def full_windows(cls, corpus: TimeSeriesCorpus):
        windows = [(i, 0, len(ts)) for i, ts in enumerate(corpus)]
        return cls(corpus, windows)

    @classmethod
    def create_dataset(
        cls,
        n_instances_train: int,
        n_instances_test: int,
        n_samples,
        *freqs,
        **kwargs,
    ):
        train_corpus = cls.load_data(
            n_instances_train, n_samples, *freqs, **kwargs
        )
        valid_corpus = cls.load_data(
            n_instances_test, n_samples, *freqs, **kwargs
        )
        test_corpus = cls.load_data(
            n_instances_test, n_samples, *freqs, **kwargs
        )
        dataset = MetaDataset(
            train_set=cls.full_windows(train_corpus),
            valid_set=cls.full_windows(valid_corpus),
            test_set=cls.full_windows(test_corpus),
        )
        return dataset


class FourierDataset(SeasonalDataset, FourierSignalReader):
    pass


class ComplexDataset(SeasonalDataset, FusedSeasonalSignalReader):
    pass


class RandomDataset(SeasonalDataset, RandomSeasonalSignalReader):
    pass


class TrendFourierDataset(SeasonalDataset, TrendFourierSignalReader):
    pass


class BenchmarkDataset(WindowsDataset, GluonTSJsonReader):
    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        window = super(BenchmarkDataset, self).__getitem__(index)
        target = window.target
        nan_mask = np.isnan(target).any(axis=1)
        nan_mask = pt.tensor(nan_mask, dtype=pt.bool)
        target = np.nan_to_num(target)
        target = pt.tensor(target, dtype=pt.float)
        feat_names = sorted(window.revealed_numerical_features.keys())
        if len(feat_names) > 0:
            feats = pt.tensor(
                np.concatenate(
                    [
                        window.revealed_numerical_features[k]
                        for k in feat_names
                    ],
                    axis=1,
                ),
                dtype=pt.float,
            )
        else:
            feats = None

        return target, feats, nan_mask

    @property
    def d_data(self) -> int:
        return 1

    @property
    def d_feats(self) -> int:
        instance = self.corpus.instances[0]
        keys = instance.revealed_numerical_features.keys()
        return sum(instance._features[k].shape[-1] for k in keys)

    @classmethod
    def create_dataset(
        cls,
        dataset: str,
        freq: str,
        n_instances_train: int,
        n_instances_eval: int,
        eval_size: int,
        window_size: int,
        window_size_plus: int = 0,
        window_size_minus: int = 0,
        window_shift: int = 1,
        train_size: Optional[int] = None,
        validation: bool = True,
        seed: int = 42,
    ) -> MetaDataset:
        corpus = cls.load_data(dataset, freq)
        train_corpus, test_corpus = corpus.split_from_end(eval_size, 0)
        if validation:
            train_corpus, valid_corpus = train_corpus.split_from_end(
                eval_size, 0, raise_error=False
            )
        else:
            valid_corpus = None
        if train_size is not None:
            _, train_corpus = train_corpus.split_from_end(
                train_size, 0, raise_error=False
            )

        kwargs = {}
        kwargs["test_set"] = cls.random_windows(
            corpus=test_corpus,
            n_windows=n_instances_eval,
            window_size=window_size,
            window_size_plus=window_size_plus,
            window_size_minus=window_size_minus,
            window_shift=window_shift,
            seed=seed,
        )
        if valid_corpus is not None:
            kwargs["valid_set"] = cls.random_windows(
                corpus=valid_corpus,
                n_windows=n_instances_eval,
                window_size=window_size,
                window_size_plus=window_size_plus,
                window_size_minus=window_size_minus,
                window_shift=window_shift,
                seed=seed,
            )
        else:
            kwargs["valid_set"] = None
        kwargs["train_set"] = cls.random_windows(
            corpus=train_corpus,
            n_windows=n_instances_train,
            window_size=window_size,
            window_size_plus=window_size_plus,
            window_size_minus=window_size_minus,
            window_shift=window_shift,
            seed=seed,
        )
        if window_size_plus > 0 or window_size_minus > 0:
            kwargs["collate_fn"] = _variable_length_collator
        return MetaDataset(**kwargs)
