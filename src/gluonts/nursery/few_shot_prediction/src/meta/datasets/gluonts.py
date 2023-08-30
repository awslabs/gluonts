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

from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
from functools import cached_property
import shutil
import tempfile
import pickle
import pandas as pd
from tqdm import tqdm
import catch22
import pytorch_lightning as pl
from gluonts.dataset.repository.datasets import materialize_dataset
from gluonts.dataset.common import MetaData
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

from meta.data.batch import TripletBatch
from meta.data.sampling import SamplingTripletDataset, SequentialTripletDataset
from meta.datasets.splits import DatasetSplits
from meta.data.dataset import TimeSeries
from meta.datasets.preprocessing import (
    Filter,
    AbsoluteValueFilter,
    ConstantTargetFilter,
    EndOfSeriesCutFilter,
    MinLengthFilter,
    read_transform_write,
    ItemIDTransform,
)
from meta.datasets.registry import register_data_module


@register_data_module
class GluonTSDataModule(pl.LightningDataModule):
    """
    A data module which provides an arbitrary dataset from GlounTS.

    Parameters
    ----------
    dataset_name: str
        The GluonTS name of the dataset.
    context_length_multiple: int
        Context length is prediction length x ``context_length_multiple``.
    support_length_multiple: int
        Support time series length is prediction length x ``support_length_multiple``.
    support_set_size: int
        Number of time series in the support set.
    num_queries: int
        Number of queries used for the same support set.
    data_dir: pathlib.Path
        Path to data.
    prediction_length: int
        Number of predicted time steps.
    standardize: bool
        If each time series is standardized.
    batch_size_train: int
        Batch size during training.
    batch_size_val_test: int
        Batch size during validation and testing.
    num_workers: int
        Number of workers.
    seed: int
        The random seed.
    catch22_train: bool
        If catch22 similarity measure is used for support set selection during training.
    catch22_val_test: bool
        If catch22 similarity measure is used for support set selection during validation and testing.
    cheat: float
        Chance of hiding the ground truth in the support set.
    """

    def __init__(
        self,
        dataset_name: str,
        context_length_multiple: int,
        support_length_multiple: int,
        support_set_size: int,
        num_queries: int = 1,
        data_dir: Path = Path.home() / ".mxnet" / "gluon-ts",
        prediction_length: Optional[int] = None,
        standardize: bool = True,
        batch_size_train: int = 128,
        batch_size_val_test: int = 1000,
        num_workers: int = 0,
        seed: Optional[int] = None,
        catch22_train: bool = False,
        catch22_val_test: bool = False,
        cheat: float = 0.0,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.context_length_multiple = context_length_multiple
        self.support_length_multiple = support_length_multiple
        self._prediction_length = prediction_length
        self.support_set_size = support_set_size
        self.num_queries = num_queries
        self.standardize = standardize
        self.splits: DatasetSplits
        self.batch_size_train = batch_size_train
        self.batch_size_val_test = batch_size_val_test
        self.num_workers = num_workers
        self.seed = seed
        self.catch22_train = catch22_train
        self.catch22_val_test = catch22_val_test
        self.cheat = cheat
        assert (
            not cheat or num_queries == 1
        ), "Cheat sampling only allows num_queries = 1"
        assert not (
            (catch22_val_test or catch22_train) and num_queries > 1
        ), "Catch22 support set selection only works with num_queries equal to one."

    @property
    def context_length(self) -> int:
        return self.context_length_multiple * self.prediction_length

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
        return self.data_dir / "datasets" / self.dataset_name

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

    def sampling_triplet_dataset(
        self, split: Literal["train", "val"]
    ) -> SamplingTripletDataset:
        if split == "train":
            dataset = self.splits.train().data()
        elif split == "val":
            dataset = self.splits.val().data()
        else:
            raise ValueError(
                "test split should not be used with sampling data set"
            )
        return SamplingTripletDataset(
            dataset=dataset,
            context_length=self.context_length,
            support_length=self.support_length,
            prediction_length=self.prediction_length,
            support_set_size=self.support_set_size,
            num_queries=self.num_queries,
            catch22_nn=self.catch22_nn if self.catch22_train else None,
            cheat=self.cheat,
        )

    def sequential_triplet_dataset(
        self,
        split: Literal["val", "test"],
    ) -> SequentialTripletDataset:
        if split == "val":
            dataset = self.splits.val().data()
            support_dataset = None
        elif split == "test":
            dataset = self.splits.test().data()
            support_dataset = self.splits.val().data()
        else:
            raise ValueError(
                "train split should not be used with sequential data set"
            )
        return SequentialTripletDataset(
            dataset=dataset,
            support_dataset=support_dataset,
            context_length=self.context_length,
            support_length=self.support_length,
            prediction_length=self.prediction_length,
            support_set_size=self.support_set_size,
            num_queries=1,
            seed=self.seed + (split == "test") if self.seed else None,
            catch22_nn=self.catch22_nn if self.catch22_val_test else None,
            cheat=self.cheat,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # a hack
        if self.dataset_name in [
            "wind_farms_without_missing",
            "kdd_cup_2018_without_missing",
            "pedestrian_counts",
        ]:
            self.catch22_tail = 50000
        else:
            self.catch22_tail = 7000

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

    @cached_property
    def catch22_nn(self):
        with open(self.root / "catch22" / "nn_100.pkl", "rb") as f:
            return pickle.load(f)

    def train_dataloader(self) -> DataLoader[TripletBatch]:
        return DataLoader(
            self.sampling_triplet_dataset("train"),
            collate_fn=TripletBatch.collate,  # type: ignore
            batch_size=self.batch_size_train,
        )

    def val_dataloader(self) -> DataLoader[TripletBatch]:
        return DataLoader(
            self.sequential_triplet_dataset("val"),
            collate_fn=TripletBatch.collate,  # type: ignore
            batch_size=self.batch_size_val_test,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[TripletBatch]:
        return DataLoader(
            self.sequential_triplet_dataset("test"),
            collate_fn=TripletBatch.collate,  # type: ignore
            batch_size=self.batch_size_val_test,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def generate(self) -> None:
        if not self.root.exists():
            self.root.mkdir(parents=True)

            # Download data and move to our own managed directory
            with tempfile.TemporaryDirectory() as directory:
                self._materialize(Path(directory))
                source = Path(directory) / self.dataset_name

                # Copy and read metadata
                meta_file = self.root / "metadata.json"
                shutil.copyfile(source / "metadata.json", meta_file)
                meta = MetaData.parse_file(meta_file)

                # Copy the data and apply filters
                # We filter all ts with length smaller than 3 * prediction length in the train / val set.
                # Since for the train split the prediction-length-long tail is cropped
                # we have that ts in train split are at least 2 * prediction length long.
                # This makes it possible to predict a full prediction length
                # and have a context window of size prediction length.
                filters = self._filters(
                    meta.prediction_length, 3 * meta.prediction_length
                )
                read_transform_write(
                    self.root / "train" / "data.json",
                    filters=filters
                    + [
                        EndOfSeriesCutFilter(meta.prediction_length),
                        ItemIDTransform(),
                    ],
                    source=source / "train" / "data.json",
                )
                num_train = count_lines(self.root / "train" / "data.json")

                read_transform_write(
                    self.root / "val" / "data.json",
                    filters=filters + [ItemIDTransform(num_train)],
                    source=source / "train" / "data.json",
                )

                # Although we increase the prediction length for the filters here, this does not
                # exclude any more data! The time series is only longer by the prediction length...
                read_transform_write(
                    self.root / "test" / "data.json",
                    filters=self._filters(
                        2 * meta.prediction_length, 4 * meta.prediction_length
                    )
                    + [
                        ItemIDTransform(num_train),
                    ],
                    source=source / "test" / "data.json",
                )
        num_train = count_lines(self.root / "train" / "data.json")
        num_val = count_lines(self.root / "val" / "data.json")
        num_test = count_lines(self.root / "test" / "data.json")

        assert num_train == num_val and (
            num_test % num_val == 0
        ), "Splits do not match."

        # compute catch22 features
        file = self.root / "catch22" / "features.pkl"
        val_data = (
            DatasetSplits(
                self.meta,
                self.root,
                self.dataset_name,
                self.prediction_length,
                _standardize=False,
            )
            .val()
            .data()
        )
        if not file.exists() and (self.catch22_train or self.catch22_val_test):
            file.parent.mkdir(parents=True, exist_ok=True)

            ts_features = []
            # running this in parallel with more workers seems to throw errors
            for ts in tqdm(val_data, desc="generate catch22 features"):
                # only use 5000 times steps tail to compute features
                start = max(len(ts) - self.catch22_tail, 0)
                ts_features.append(get_features(ts[start:]))

            df = pd.DataFrame(ts_features)
            # remove constant features
            df = df.loc[:, (df != df.iloc[0]).any()]
            df.to_pickle(file)

            # compute catch22 nearest neighbours
        file = self.root / "catch22" / "nn_100.pkl"
        if not file.exists() and (self.catch22_train or self.catch22_val_test):
            df = normalize_features(self.catch22())
            num_ts = len(val_data)
            # for more than 5000 ts use kd_tree algo
            algo = "brute" if num_ts <= 5000 else "kd_tree"
            num_nn = 100
            nbrs = NearestNeighbors(
                n_neighbors=min(num_ts, num_nn + 1), algorithm=algo
            ).fit(df)
            distances, indices = nbrs.kneighbors(df)
            with open(file, "wb") as f:
                # exclude the query ts itself
                pickle.dump(indices[:, 1:], f)

    def catch22(self) -> pd.DataFrame:
        """
        Returns the catch22 features of all time series in the dataset.
        """
        file = self.root / "catch22" / "features.pkl"
        return pd.read_pickle(file)

    @classmethod
    def name(cls) -> str:
        return "dm_gluonts"

    def evaluate_model(self, **kwargs):
        pass

    def _filters(
        self, prediction_length: int, min_length: int
    ) -> List[Filter]:
        return [
            ConstantTargetFilter(
                prediction_length, required_length=self.catch22_tail
            ),
            AbsoluteValueFilter(1e18),
            MinLengthFilter(min_length),
        ]

    def _materialize(self, directory: Path) -> None:
        materialize_dataset(self.dataset_name, directory)


def get_features(ts: TimeSeries) -> Dict[str, Any]:
    features = catch22.catch22_all(ts.values)
    return dict(zip(features["names"], features["values"]))


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def count_lines(filename: Path) -> int:
    return sum(1 for line in open(filename))
