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

from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple
from pathlib import Path
from tqdm.auto import tqdm
import json
import numpy as np
import random
import pytorch_lightning as pl
from gluonts.dataset.field_names import FieldName  # type: ignore
from gluonts.dataset.repository.datasets import get_dataset, materialize_dataset  # type: ignore
from torch.utils.data import DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import warnings

warnings.filterwarnings("ignore")

from meta.data.batch import TripletBatch
from meta.datasets.splits import DatasetSplits
from meta.data.sampling import TripletDataset
from meta.datasets.registry import register_data_module


@dataclass
class MetaData:
    """
    Meta data for artificial datasets
    """

    context_length_multiple: int
    support_set_size: int
    prediction_length: int
    freq: str
    support_length_multiple: int = 4
    num_queries: int = 1

    @classmethod
    def parse_file(cls, file: Path):
        with open(file) as json_file:
            return MetaData(**json.load(json_file))

    def save(self, file: Path):
        with open(file, "w") as fp:
            json.dump(asdict(self), fp)


@register_data_module
class ArtificialDataModule(pl.LightningDataModule):
    """
    A data module which provides an artificial dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        context_length_multiple: int,
        support_length_multiple: int,
        support_set_size: int,
        num_queries: int = 1,
        data_dir: Path = Path.home() / ".mxnet" / "gluont-ts",
        prediction_length: Optional[int] = None,
        test_set_size: int = 200,
        val_set_size: int = 200,
        train_set_size: int = 10000,
        standardize: bool = False,
        batch_size_train: int = 128,
        batch_size_val: int = 1000,
        batch_size_test: int = 1000,
        num_workers: int = 0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.context_length_multiple = context_length_multiple
        self.support_length_multiple = support_length_multiple
        self._prediction_length = prediction_length
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size
        self.train_set_size = train_set_size
        self.support_set_size = support_set_size
        self.num_queries = num_queries
        self.standardize = standardize
        self.splits: DatasetSplits
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.seed = seed

    @property
    def context_length(self) -> int:
        return self.context_length_multiple * self.meta.prediction_length

    @property
    def support_length(self) -> int:
        return self.support_length_multiple * self.prediction_length

    @property
    def prediction_length(self) -> int:
        return self._prediction_length or self.meta.prediction_length

    @property
    def root(self) -> Path:
        """
        Returns the directory where all the data pertaining to this dataset is stored.
        """
        return self.data_dir / "artificial" / self.dataset_name

    @property
    def meta(self) -> MetaData:
        """
        Returns the dataset's metadata.
        """
        return (
            MetaData.parse_file(self.root / "metadata.json")
            if self.root.exists()
            else None
        )

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
        self.splits.val().prepare()
        self.splits.test().prepare()

    def train_dataloader(self) -> DataLoader[TripletBatch]:
        split = self.splits.train()
        return DataLoader(
            TripletDataset(
                queries=split.data(), support_sets=split.support_set()
            ),
            collate_fn=TripletBatch.collate,  # type: ignore
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[TripletBatch]:
        split = self.splits.val()
        return DataLoader(
            TripletDataset(
                queries=split.data(), support_sets=split.support_set()
            ),
            collate_fn=TripletBatch.collate,  # type: ignore
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[TripletBatch]:
        split = self.splits.train()
        return DataLoader(
            TripletDataset(
                queries=split.data(), support_sets=split.support_set()
            ),
            collate_fn=TripletBatch.collate,  # type: ignore
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def generate(self) -> None:
        meta = MetaData(
            context_length_multiple=self.context_length_multiple,
            support_length_multiple=self.support_length_multiple,
            support_set_size=self.support_set_size,
            prediction_length=self.prediction_length,
            num_queries=self.num_queries,
            freq="1H",
        )
        if self.root.exists():
            # check if the meta data of the dataset fits the requirements
            if self.meta == meta:
                return
            else:
                raise ValueError(
                    "Meta data of artificial dataset not compatible with requirements"
                )

        self.root.mkdir(parents=True, exist_ok=True)
        meta.save(self.root / "metadata.json")
        for split, n_samples in zip(
            ["train", "val", "test"],
            [self.train_set_size, self.val_set_size, self.test_set_size],
        ):
            self.generate_split(split, n_samples)

    def generate_split(
        self, split: Literal["train", "val", "test"], n_samples: int
    ) -> None:
        queries, support_sets = zip(
            *[
                generate_artificial_tuplets(
                    dataset_name=self.dataset_name,
                    context_length=self.context_length,
                    support_length=self.support_length,
                    prediction_length=self.prediction_length,
                    support_set_size=self.support_set_size,
                    item_id=i,
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

    @classmethod
    def name(cls) -> str:
        return "dm_artificial"

    def evaluate_model(self, **kwargs):
        pass


def _write_data_to_file(file: Path, data: Tuple) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w") as f:
        content = "\n".join([json.dumps(d) for d in data])
        f.write(content + "\n")


def generate_artificial_tuplets(
    dataset_name: str,
    context_length: int,
    support_length: int,
    prediction_length: int,
    support_set_size: int,
    item_id: int,
) -> Tuple:
    if dataset_name == "marker":
        noise_level = 0.01
        context = np.random.normal(0, noise_level, context_length)
        signal_type = random.choice([-1, 1])
        target = signal_type * np.arange(0, 1, 1 / prediction_length)
        query = np.concatenate((context, target))

        # support time series without the marker
        support_set = list(
            np.random.normal(0, noise_level, context_length)
            for _ in range(support_set_size - 1)
        )
        # build support time series which contains the marker
        marker = (
            (-1)
            * signal_type
            * np.concatenate(
                (np.arange(0, 1, 1 / 6), np.flip(np.arange(0, 1, 1 / 6)))
            )
        )
        marker_start = np.random.choice(
            np.arange(0, context_length - len(marker))
        )
        support_ts = np.random.normal(0, noise_level, context_length)
        support_ts[marker_start : marker_start + len(marker)] += marker
        # insert at random position in support set
        support_set.insert(np.random.choice(support_set_size), support_ts)

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

    if dataset_name == "marker_v2":
        noise_level = 0.01
        context = np.random.normal(0, noise_level, context_length)
        signal_type = random.choice([-1, 1])
        target = signal_type * np.arange(0, 1, 1 / prediction_length)
        query = np.concatenate((context, target))

        # support time series without the marker
        support_set = list(
            np.random.normal(0, noise_level, context_length)
            for _ in range(support_set_size - 1)
        )
        # build support time series which contains the marker
        ramp = np.arange(0, 1, 1 / 6)
        if signal_type == -1:
            marker = np.concatenate((ramp, np.flip(ramp)))
        else:
            marker = np.concatenate(
                (ramp, np.array([1, 0.8, 1]), np.flip(ramp))
            )
        marker_start = np.random.choice(
            np.arange(0, context_length - len(marker))
        )
        support_ts = np.random.normal(0, noise_level, context_length)
        support_ts[marker_start : marker_start + len(marker)] += marker
        # insert at random position in support set
        support_set.insert(np.random.choice(support_set_size), support_ts)

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
    if dataset_name == "marker_v3":
        noise_level = 0.01
        context = np.random.normal(0, noise_level, context_length)
        signal_type = random.choice([-1, 0, 1])
        target = signal_type * np.arange(0, 1, 1 / prediction_length)
        query = np.concatenate((context, target))

        # support time series without the marker
        support_set = list(
            np.random.normal(0, noise_level, context_length)
            for _ in range(support_set_size - 1)
        )
        # build support time series which contains the marker
        ramp = np.arange(0, 1, 1 / 6)
        if signal_type == -1:
            marker = np.concatenate((ramp, np.flip(ramp)))
        elif signal_type == 1:
            marker = np.concatenate(
                (ramp, np.array([1, 0.8, 1]), np.flip(ramp))
            )
        else:
            marker = np.concatenate(
                (ramp, np.array([1, 1, 1, 1, 1]), np.flip(ramp))
            )
        marker_start = np.random.choice(
            np.arange(0, context_length - len(marker))
        )
        support_ts = np.random.normal(0, noise_level, context_length)
        support_ts[marker_start : marker_start + len(marker)] += marker
        # insert at random position in support set
        support_set.insert(np.random.choice(support_set_size), support_ts)

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
    if dataset_name == "marker_v4":
        noise_level = 0.01
        context = np.random.normal(0, noise_level, context_length)
        signal_type = random.choice([-1, 0, 1])
        target = signal_type * np.arange(0, 1, 1 / prediction_length)
        query = np.concatenate((context, target))

        # support time series without the marker
        support_set = list(
            np.random.normal(0, noise_level, context_length)
            for _ in range(support_set_size - 1)
        )
        # build support time series which contains the marker
        ramp = np.arange(0, 1, 1 / 3)
        if signal_type == -1:
            marker = np.concatenate((ramp, np.flip(ramp)))
        elif signal_type == 1:
            marker = np.concatenate(
                (ramp, np.array([1, 0.8, 1]), np.flip(ramp))
            )
        else:
            marker = np.concatenate((ramp, np.array([1, 1, 1]), np.flip(ramp)))
        marker_start = np.random.choice(
            np.arange(0, context_length - len(marker))
        )
        support_ts = np.random.normal(0, noise_level, context_length)
        support_ts[marker_start : marker_start + len(marker)] += marker
        # insert at random position in support set
        support_set.insert(np.random.choice(support_set_size), support_ts)

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
    if dataset_name == "marker_v6":
        noise_level = 0.01
        context = np.random.normal(0, noise_level, context_length)
        signal_type = random.choice([-1, 0, 1])
        target = signal_type * np.arange(0, 1, 1 / prediction_length)
        query = np.concatenate((context, target))

        # support time series without the marker
        support_set = list(
            np.random.normal(0, noise_level, context_length)
            for _ in range(support_set_size - 1)
        )
        # build support time series which contains the marker
        markers = [4 * [0.2], [0, 0.2, 0.2, 0], [0, 0.15, 0.2, 0.25]]
        marker = markers[signal_type + 1]
        marker_start = np.random.choice(
            np.arange(0, context_length - len(marker))
        )
        support_ts = np.random.normal(0, noise_level, context_length)
        support_ts[marker_start : marker_start + len(marker)] = marker
        # insert at random position in support set
        support_set.insert(np.random.choice(support_set_size), support_ts)
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
    if dataset_name == "cheat_random":
        context = np.random.normal(0, 0.001, context_length)
        target = np.random.normal(0, 0.1, prediction_length)
        query = np.concatenate((context, target))
        support_set = list(
            np.random.normal(0, 0.001, context_length)
            for _ in range(support_set_size)
        )
        si = np.random.choice(support_set_size)
        support_set[si][-prediction_length:] = query[-prediction_length:]
        # insert at random position in support set
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
    if dataset_name == "cheat_random_gp":
        kernel = 0.5 * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 10.0))
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=item_id)
        query = gpr.sample_y(
            np.arange(prediction_length + context_length).reshape(-1, 1),
            1,
            random_state=item_id + np.random.randint(100000),
        ).squeeze()

        marker_length = prediction_length

        support_set = gpr.sample_y(
            np.arange(support_length).reshape(-1, 1),
            support_set_size,
            random_state=item_id + np.random.randint(100000),
        ).T
        signal_length = marker_length + prediction_length
        marker_start = np.random.choice(
            np.arange(0, support_length - signal_length)
        )
        si = np.random.choice(support_set_size)
        support_set[si][marker_start : marker_start + signal_length] = query[
            -signal_length:
        ]

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
    if dataset_name == "cheat_random_gp_fitted":
        kernel = 0.5 * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 10.0))
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=item_id)
        query = gpr.sample_y(
            np.arange(prediction_length + context_length).reshape(-1, 1),
            1,
            random_state=item_id + np.random.randint(100000),
        ).squeeze()

        marker_length = prediction_length

        support_set = gpr.sample_y(
            np.arange(context_length).reshape(-1, 1),
            support_set_size,
            random_state=item_id + np.random.randint(100000),
        ).T
        signal_length = marker_length + prediction_length
        marker_start = np.random.choice(
            np.arange(0, context_length - signal_length)
        )
        si = np.random.choice(support_set_size)
        x = np.arange(marker_start, marker_start + signal_length)
        gpr.fit(x.reshape(-1, 1), query[-signal_length:].reshape(-1, 1))
        s_ts = gpr.sample_y(
            np.arange(context_length).reshape(-1, 1),
            1,
            random_state=item_id + np.random.randint(100000),
        )
        support_set[si] = s_ts.squeeze()

        # insert at random position in support set
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
    if dataset_name == "cheat_random_gp_mix_marker":
        signal_type = random.choice(["gp", "marker"])
        kernel = 0.5 * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 10.0))
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=item_id)
        query = gpr.sample_y(
            np.arange(prediction_length + context_length).reshape(-1, 1),
            1,
            random_state=item_id + np.random.randint(100000),
        ).squeeze()
        support_set = gpr.sample_y(
            np.arange(context_length).reshape(-1, 1),
            support_set_size,
            random_state=item_id + np.random.randint(100000),
        ).T
        if signal_type == "gp":
            marker_length = prediction_length
            signal_length = marker_length + prediction_length
            marker_start = np.random.choice(
                np.arange(0, context_length - signal_length)
            )
            si = np.random.choice(support_set_size)
            support_set[si][
                marker_start : marker_start + signal_length
            ] = query[-signal_length:]
        else:
            signal = np.concatenate(
                (np.ones((4,)), query[-prediction_length:])
            )
            signal_length = prediction_length + 4
            marker_start = np.random.choice(
                np.arange(0, context_length - signal_length)
            )
            si = np.random.choice(support_set_size)

            support_set[si][
                marker_start : marker_start + signal_length
            ] = signal
        # insert at random position in support set
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
    if dataset_name == "cheat_random_gp_mix":
        signal_type = random.choices(
            ["gp", "marker"], weights=[0.5, 0.5], k=1
        )[0]
        kernel = 0.5 * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 10.0))
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=item_id)
        query = gpr.sample_y(
            np.arange(prediction_length + context_length).reshape(-1, 1),
            1,
            random_state=item_id + np.random.randint(100000),
        ).squeeze()
        support_set = gpr.sample_y(
            np.arange(context_length).reshape(-1, 1),
            support_set_size,
            random_state=item_id + np.random.randint(100000),
        ).T
        if signal_type == "gp":
            marker_length = prediction_length
            signal_length = marker_length + prediction_length
            marker_start = np.random.choice(
                np.arange(0, context_length - signal_length)
            )
            si = np.random.choice(support_set_size)
            support_set[si][
                marker_start : marker_start + signal_length
            ] = query[-signal_length:]
        # else:
        #     signal = query[-prediction_length:]
        #     signal_length = prediction_length
        #     marker_start = np.random.choice(np.arange(0, context_length - signal_length))
        #     si = np.random.choice(support_set_size)

        #     support_set[si][marker_start : marker_start + signal_length] = signal
        # insert at random position in support set
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
    if dataset_name == "random_gp":
        kernel = 0.5 * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 10.0))
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=item_id)
        query = gpr.sample_y(
            np.arange(prediction_length + context_length).reshape(-1, 1),
            1,
            random_state=item_id + np.random.randint(100000),
        ).squeeze()

        support_set = gpr.sample_y(
            np.arange(context_length).reshape(-1, 1),
            support_set_size,
            random_state=item_id + np.random.randint(100000),
        ).T

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
