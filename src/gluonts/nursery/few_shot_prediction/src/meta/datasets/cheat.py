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

import os
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple, List, Dict
from pathlib import Path
import json
from tqdm import tqdm
import hashlib
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from torch.utils.data import DataLoader, ConcatDataset
from gluonts.time_feature import get_seasonality

from .artificial import _write_data_to_file
from meta.datasets.registry import register_data_module
from meta.data.batch import TripletBatch
from meta.datasets.splits import DatasetSplits
from meta.data.sampling import TripletDataset
from meta.common.torch import tensor_to_np
from meta.metrics.numpy import compute_metrics


@dataclass(frozen=True)
class CheatMetaData:
    """
    Meta data for a cheating dataset
    """

    context_length_multiple: int
    support_length_multiple: int
    prediction_length: int
    test_set_size: int
    val_set_size: int
    train_set_size: int
    freq: str
    # below are the parameters that we will play in the artificial experiments
    support_set_size: int
    cheat_chance: float
    noise_level: float
    query_length_scale: float
    support_length_scale_same_as_query: bool

    @classmethod
    def parse_file(cls, file: Path):
        with open(file) as json_file:
            return CheatMetaData(**json.load(json_file))

    def save(self, file: Path):
        with open(file, "w") as fp:
            json.dump(asdict(self), fp)

    def get_hash(self):
        return (
            hashlib.md5(json.dumps(asdict(self)).encode("utf-8"))
            .digest()
            .hex()
        )


@register_data_module
class CheatArtificialDataModule(pl.LightningDataModule):
    """
    A data module which provides datasets with different
    ground truth / counter factual injection modes for the support set.

    Parameters
    ----------
    context_length_multiple: int
        Context length is prediction length x ``context_length_multiple``.
    prediction_length: int
        Number of predicted time steps.
    support_length_multiple: int
        Support time series length is prediction length x ``support_length_multiple``.
    test_set_size: int
        The size of the test split of the generated dataset.
    val_set_size: int
        The size of the val split of the generated dataset.
    train_set_size: int
        The size of the train split of the generated dataset.
    freq: str
        Not clear if this is actually used here on not.
    support_set_size: int
        Number of time series in the support set.
    data_dir: pathlib.Path
        Path to data.
    cheat_chance: float
        The chance of injecting ground truth in a support set.
    noise_level: float
        The level of noise added to the ground truth when cheating.
    query_length_scale: float
        The length scale used by the gaussian process to sample the queries from.
    support_length_scale_same_as_query: bool
        Whether the length scale of the gaussian process for query and support set match.
    standardize: bool
        If each time series is standardized.
    batch_size_train: int
        Batch size during training.
    batch_size_val_test: int
        Batch size during validation and testing.
    num_workers: int
        Number of workers.
    random_seed: int
        The random seed.
    """

    def __init__(
        self,
        # dataset metadata config starts
        context_length_multiple: int = 4,
        prediction_length: int = 24,
        support_length_multiple: int = 4,
        test_set_size: int = 500,
        val_set_size: int = 500,
        train_set_size: int = 10000,
        freq: str = "H",
        support_set_size: int = 3,
        cheat_chance: float = 1.0,
        noise_level: float = 0.0,
        query_length_scale: float = 5.0,
        support_length_scale_same_as_query: bool = True,
        # data loader config starts
        data_dir: Path = Path.home() / ".mxnet" / "gluon-ts",
        standardize: bool = False,
        batch_size_train: int = 128,
        batch_size_val_test: int = 1000,
        num_workers: int = 0,
        random_seed: int = int.from_bytes(os.urandom(3), byteorder="big"),
    ):
        assert 0.1 <= query_length_scale <= 10

        super().__init__()
        # parameters of the dataset
        self.meta = CheatMetaData(
            context_length_multiple=context_length_multiple,
            prediction_length=prediction_length,
            support_length_multiple=support_length_multiple,
            test_set_size=test_set_size,
            val_set_size=val_set_size,
            train_set_size=train_set_size,
            freq=freq,
            support_set_size=support_set_size,
            cheat_chance=cheat_chance,
            noise_level=noise_level,
            query_length_scale=query_length_scale,
            support_length_scale_same_as_query=support_length_scale_same_as_query,
        )
        # parameters of the loader
        self.data_dir = data_dir
        self.standardize = standardize
        self.batch_size_train = batch_size_train
        self.batch_size_val_test = batch_size_val_test
        self.num_workers = num_workers
        self.splits: DatasetSplits
        self.random_state = np.random.RandomState(random_seed)

    @property
    def dataset_names_val_test(self) -> List[str]:
        return list({"cc1.0", f"cc{self.meta.cheat_chance}"})

    @property
    def dataset_names_val(self) -> List[str]:
        return self.dataset_names_val_test

    @property
    def dataset_names_test(self) -> List[str]:
        return self.dataset_names_val_test

    @property
    def context_length(self) -> int:
        return self.meta.context_length_multiple * self.meta.prediction_length

    @property
    def support_length(self) -> int:
        return self.meta.support_length_multiple * self.meta.prediction_length

    @property
    def prediction_length(self) -> int:
        return self.meta.prediction_length

    @property
    def dataset_name(self) -> str:
        return "cheat_" + self.meta.get_hash()

    @property
    def root(self) -> Path:
        """
        Returns the directory where all the data pertaining to this dataset is stored.
        """
        return self.data_dir / "artificial" / self.dataset_name

    def setup(self, stage: Optional[str] = None) -> None:
        self.generate()
        self.splits = DatasetSplits(
            self.meta,
            self.root,
            self.dataset_name,
            self.prediction_length,
            self.standardize,
        )
        self.splits.train().prepare()
        for d_name in self.dataset_names_val:
            self.splits.val(d_name).prepare()
        for d_name in self.dataset_names_test:
            self.splits.test(d_name).prepare()

    def train_dataloader(self) -> DataLoader[TripletBatch]:
        split = self.splits.train()
        return DataLoader(
            TripletDataset(
                queries=split.data(), support_sets=split.support_set()
            ),
            collate_fn=TripletBatch.collate,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[TripletBatch]:
        splits = [self.splits.val(d_name) for d_name in self.dataset_names_val]

        return list(
            [
                DataLoader(
                    TripletDataset(
                        queries=split.data(), support_sets=split.support_set()
                    ),
                    collate_fn=TripletBatch.collate,
                    batch_size=self.batch_size_val_test,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                for split in splits
            ]
        )

    def test_dataloader(self) -> DataLoader[TripletBatch]:
        splits = [
            self.splits.test(d_name) for d_name in self.dataset_names_test
        ]
        return list(
            [
                DataLoader(
                    TripletDataset(
                        queries=split.data(), support_sets=split.support_set()
                    ),
                    collate_fn=TripletBatch.collate,
                    batch_size=self.batch_size_val_test,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                for split in splits
            ]
        )

    def generate(self) -> None:
        if self.root.exists():
            return

        self.root.mkdir(parents=True, exist_ok=True)
        self.meta.save(self.root / "metadata.json")
        for split, n_samples in zip(
            ["train", "val", "test"],
            [
                self.meta.train_set_size,
                self.meta.val_set_size,
                self.meta.test_set_size,
            ],
        ):
            if split in ["val", "test"]:
                self.generate_split(
                    split + "_cc1.0", n_samples, always_cheat=True
                )
                self.generate_split(
                    split + f"_cc{self.meta.cheat_chance}",
                    n_samples,
                    always_cheat=False,
                )
            else:
                self.generate_split(split, n_samples, always_cheat=False)

    def generate_split(
        self,
        split: Literal["train", "val", "test"],
        n_samples: int,
        always_cheat=True,
        query_length_scale=5.0,
    ) -> None:
        queries, support_sets = zip(
            *[
                self.generate_artificial_tuplets(
                    item_id=i,
                    always_cheat=always_cheat,
                    query_length_scale=query_length_scale,
                )
                for i in tqdm(
                    range(n_samples), desc="generating artificial data"
                )
            ]
        )
        _write_data_to_file(self.root / split / "data.json", queries)
        _write_data_to_file(
            self.root / split / ".support_set.json", support_sets
        )

    def generate_artificial_tuplets(
        self, item_id: int, always_cheat: bool, query_length_scale: float
    ) -> Tuple:
        # generate query
        query_length = self.meta.prediction_length + self.context_length
        query_x = np.arange(query_length).reshape(-1, 1)
        support_x = np.arange(self.support_length).reshape(-1, 1)

        kernel = 0.5 * RBF(
            length_scale=query_length_scale, length_scale_bounds=(1e-1, 10.0)
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel, random_state=self.random_state
        )
        query = gpr.sample_y(
            query_x, 1, random_state=self.random_state
        ).squeeze()
        assert query.shape == (query_length,), query.shape

        # prepare support set
        if self.meta.support_length_scale_same_as_query:
            support_set = gpr.sample_y(
                support_x,
                self.meta.support_set_size,
                random_state=self.random_state,
            ).T
        else:
            ys = []
            for _ in range(self.meta.support_set_size):
                sample_length_scale = self.random_state.uniform(0, 10)
                kernel = 0.5 * RBF(
                    length_scale=sample_length_scale,
                    length_scale_bounds=(1e-1, 10.0),
                )
                gpr_support = GaussianProcessRegressor(
                    kernel=kernel, random_state=self.random_state
                )
                y_hat = gpr_support.sample_y(support_x)
                ys.append(y_hat.ravel())
            support_set = np.array(ys)

        assert support_set.shape == (
            self.meta.support_set_size,
            self.support_length,
        ), support_set.shape

        # decide to cheat on 1 TS in support set or not based on cheat_chance
        is_cheat = (
            True
            if always_cheat
            else self.random_state.rand() <= self.meta.cheat_chance
        )

        if is_cheat:
            # generate a random place to for TS in support set to start cheating
            marker_length = self.meta.prediction_length
            signal_length = marker_length + self.meta.prediction_length
            marker_start = self.random_state.choice(
                np.arange(0, self.support_length - signal_length)
            )

            # prepare data to train the GP
            x = np.array(
                range(marker_start, marker_start + signal_length)
            ).reshape(-1, 1)
            y = query.tolist()[-signal_length:]

            # fit gp and sample from fitted gp
            gpr.fit(x, y)

            # generate a random index from support set
            si = self.random_state.choice(self.meta.support_set_size)

            if self.meta.support_length_scale_same_as_query:
                y_hat = gpr.sample_y(support_x)
            else:
                sample_length_scale = self.random_state.uniform(0, 10)
                gpr.kernel_.set_params(
                    **{"k2__length_scale": sample_length_scale}
                )
                y_hat = gpr.sample_y(support_x)
            support_set[si] = y_hat.ravel()

        # add noise
        if self.meta.noise_level > 0:
            noise = (
                self.random_state.randn(query.shape[0]) * self.meta.noise_level
            )
            query += noise

            new_support_set = []
            for s in support_set:
                noise = (
                    self.random_state.randn(s.shape[0]) * self.meta.noise_level
                )
                new_support_set.append(s + noise)
            support_set = np.array(new_support_set)

        q = {
            "target": query.tolist(),
            "item_id": item_id,
            "start": "2012-01-01 00:00:00",
        }
        s = [
            {
                "target": s.tolist(),
                "start": "2012-01-01 00:00:00",
                "item_id": item_id,
            }
            for s in support_set
        ]
        return q, s

    def get_log_batches(self, n_logging_samples: int) -> Tuple[TripletBatch]:
        log_batch_train = (
            next(iter(self.train_dataloader()))
            .reduce_to_unique_query()
            .first_n(n_logging_samples)
        )
        log_batch_val = (
            next(iter(self.val_dataloader()[0]))
            .reduce_to_unique_query()
            .first_n(n_logging_samples)
        )
        return log_batch_train, log_batch_val

    def evaluate_model(
        self,
        model: nn.Module,
        quantiles: List[str],
        test: bool = False,
    ) -> List[Dict[str, float]]:
        metrics = []

        self.setup()
        pred = []
        if test:
            split = self.splits.test()
            dl = self.test_dataloader()
        else:
            split = self.splits.val()
            dl = self.val_dataloader()

        for batch in tqdm(
            dl,
            desc=f"generating predictions for dataset {self.name()} with support set size",
        ):
            pred.append(model(supps=batch.support_set, query=batch.query_past))

        # redo standardization for evaluation
        pred = split.data().rescale_dataset(torch.cat(pred, dim=0))
        pred = tensor_to_np(pred)

        # use only the length that should be included for evaluation
        pred = pred[:, : self.prediction_length, ...]

        m = compute_metrics(
            pred,
            split.evaluation(),
            quantiles=quantiles,
            seasonality=get_seasonality(self.meta.freq),
        )
        metrics.append(m)
        return metrics

    @classmethod
    def name(cls) -> str:
        return "dm_cheat"


@dataclass(frozen=True)
class CheatCounterfactual(CheatMetaData):
    """
    Meta data for a cheating dataset
    """

    counterfactual_size: int
    counterfactual_mixing: bool

    @classmethod
    def parse_file(cls, file: Path):
        with open(file) as json_file:
            return CheatCounterfactual(**json.load(json_file))


@register_data_module
class CheatCounterfactualArtificialDataModule(CheatArtificialDataModule):
    def __init__(
        self,
        # dataset metadata config starts
        context_length_multiple: int = 4,
        prediction_length: int = 24,
        support_length_multiple: int = 4,
        test_set_size: int = 500,
        val_set_size: int = 500,
        train_set_size: int = 10000,
        freq: str = "H",
        support_set_size: int = 3,
        cheat_chance: float = 1.0,
        noise_level: float = 0.0,
        query_length_scale: float = 5.0,
        support_length_scale_same_as_query: bool = True,
        counterfactual_size: int = 1,
        counterfactual_mixing: bool = False,
        # data loader config starts
        data_dir: Path = Path.home() / ".mxnet" / "gluon-ts",
        standardize: bool = False,
        batch_size_train: int = 128,
        batch_size_val_test: int = 1000,
        num_workers: int = 0,
        random_seed: int = int.from_bytes(os.urandom(3), byteorder="big"),
    ):
        assert 0.1 <= query_length_scale <= 10
        assert counterfactual_size < support_set_size
        super().__init__()

        # parameters of the dataset
        self.meta = CheatCounterfactual(
            context_length_multiple=context_length_multiple,
            prediction_length=prediction_length,
            support_length_multiple=support_length_multiple,
            test_set_size=test_set_size,
            val_set_size=val_set_size,
            train_set_size=train_set_size,
            freq=freq,
            support_set_size=support_set_size,
            cheat_chance=cheat_chance,
            noise_level=noise_level,
            query_length_scale=query_length_scale,
            support_length_scale_same_as_query=support_length_scale_same_as_query,
            counterfactual_size=counterfactual_size,
            counterfactual_mixing=counterfactual_mixing,
        )

        # parameters of the loader
        self.data_dir = data_dir
        self.standardize = standardize
        self.batch_size_train = batch_size_train
        self.batch_size_val_test = batch_size_val_test
        self.num_workers = num_workers
        self.splits: DatasetSplits
        self.random_state = np.random.RandomState(random_seed)

    @property
    def dataset_names_val_test(self) -> List[str]:
        return list({"cc1.0", f"cf{self.meta.counterfactual_size}"})

    @property
    def dataset_names_train(self) -> List[str]:
        if self.meta.counterfactual_size == 0:
            return ["cc1.0"]
        else:
            if self.meta.counterfactual_mixing:
                return ["cc1.0", f"cf{self.meta.counterfactual_size}"]
            else:
                return [f"cf{self.meta.counterfactual_size}"]

    @property
    def dataset_name(self) -> str:
        return "cf_" + self.meta.get_hash()

    def train_dataloader(self) -> DataLoader[TripletBatch]:
        splits = [
            self.splits.train(name=d_name)
            for d_name in self.dataset_names_train
        ]

        datasets = ConcatDataset(
            [
                TripletDataset(
                    queries=split.data(), support_sets=split.support_set()
                )
                for split in splits
            ]
        )

        return DataLoader(
            datasets,
            shuffle=True,
            collate_fn=TripletBatch.collate,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def generate(self) -> None:
        if self.root.exists():
            return

        self.root.mkdir(parents=True, exist_ok=True)
        self.meta.save(self.root / "metadata.json")
        for split, n_samples in zip(
            ["train", "val", "test"],
            [
                self.meta.train_set_size,
                self.meta.val_set_size,
                self.meta.test_set_size,
            ],
        ):
            self.generate_split(
                split + "_cc1.0",
                n_samples,
                always_cheat=True,
                counterfactual_size=0,
            )
            self.generate_split(
                f"{split}_cf{self.meta.counterfactual_size}",
                n_samples,
                always_cheat=True,
                counterfactual_size=self.meta.counterfactual_size,
            )

    def generate_split(
        self,
        split: str,
        n_samples: int,
        always_cheat=True,
        query_length_scale: float = 5.0,
        counterfactual_size: int = 0,
    ) -> None:
        queries, support_sets = zip(
            *[
                self.generate_artificial_tuplets(
                    item_id=i,
                    always_cheat=always_cheat,
                    query_length_scale=query_length_scale,
                    counterfactual_size=counterfactual_size,
                )
                for i in tqdm(
                    range(n_samples), desc="generating artificial data"
                )
            ]
        )
        _write_data_to_file(self.root / split / "data.json", queries)
        _write_data_to_file(
            self.root / split / ".support_set.json", support_sets
        )

    def generate_artificial_tuplets(
        self,
        item_id: int,
        always_cheat: bool,
        query_length_scale: float,
        counterfactual_size: int,
    ) -> Tuple:
        assert always_cheat is True, "In CounterFactual, we always cheat"
        assert self.meta.support_length_scale_same_as_query is True

        # generate query
        query_length = self.meta.prediction_length + self.context_length
        query_x = np.arange(query_length).reshape(-1, 1)
        support_x = np.arange(self.support_length).reshape(-1, 1)

        kernel = 0.5 * RBF(
            length_scale=query_length_scale, length_scale_bounds=(1e-1, 10.0)
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel, random_state=self.random_state
        )
        query = gpr.sample_y(
            query_x, 1, random_state=self.random_state
        ).squeeze()
        assert query.shape == (query_length,), query.shape

        support_set = gpr.sample_y(
            support_x,
            self.meta.support_set_size,
            random_state=self.random_state,
        ).T

        assert support_set.shape == (
            self.meta.support_set_size,
            self.support_length,
        ), support_set.shape

        # generate a random place to for TS in support set to start cheating
        marker_length = self.meta.prediction_length
        signal_length = marker_length + self.meta.prediction_length
        marker_start = self.random_state.choice(
            np.arange(0, self.support_length - signal_length)
        )

        # prepare data to train the GP
        x = np.array(
            range(marker_start, marker_start + signal_length)
        ).reshape(-1, 1)
        y = query.tolist()[-signal_length:]

        # fit gp and sample from fitted gp
        gpr.fit(x, y)

        # generate a random index from support set to cheat
        si = self.random_state.choice(self.meta.support_set_size)
        y_hat = gpr.sample_y(support_x, 1, random_state=self.random_state)
        support_set[si] = y_hat.ravel()

        indices = list(range(self.meta.support_set_size))
        indices.remove(si)

        # generate a counterfactual TS, only contains information in the context (marker length)
        x_cf = np.array(
            range(marker_start, marker_start + marker_length)
        ).reshape(-1, 1)
        y_cf = query.tolist()[-signal_length:-marker_length]

        # fit gp and sample from fitted gp
        for _ in range(counterfactual_size):
            gpr_cf = GaussianProcessRegressor(
                kernel=kernel, random_state=self.random_state
            )
            gpr_cf.fit(x_cf, y_cf)
            # generate a random index from support set for cf
            cf_si = self.random_state.choice(indices)
            y_hat_cf = gpr_cf.sample_y(
                support_x, 1, random_state=self.random_state
            )
            support_set[cf_si] = y_hat_cf.ravel()
            indices.remove(cf_si)

        # add noise
        if self.meta.noise_level > 0:
            noise = (
                self.random_state.randn(query.shape[0]) * self.meta.noise_level
            )
            query += noise

            new_support_set = []
            for s in support_set:
                noise = (
                    self.random_state.randn(s.shape[0]) * self.meta.noise_level
                )
                new_support_set.append(s + noise)
            support_set = np.array(new_support_set)

        q = {
            "target": query.tolist(),
            "item_id": item_id,
            "start": "2012-01-01 00:00:00",
        }
        s = [
            {
                "target": s.tolist(),
                "start": "2012-01-01 00:00:00",
                "item_id": item_id,
            }
            for s in support_set
        ]
        return q, s

    @classmethod
    def name(cls) -> str:
        return "dm_cheat_counterfactual"
