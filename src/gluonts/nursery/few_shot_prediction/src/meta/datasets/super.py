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

from typing import Optional, Tuple
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm

from meta.data.batch import TripletBatch
from meta.data.sampling import SuperSamplingTripletDataset
from meta.datasets.registry import register_data_module
from meta.datasets import get_data_module


@register_data_module
class SuperDataModule(pl.LightningDataModule):
    """
    A data module which provides a collection of GluonTS datasets.

     Parameters
    ----------

    dataset_names_train: str
        Names of the GluonTS datasets used for training.
    dataset_names_val: str
        Names of the GluonTS datasets used for validation.
    dataset_names_test: str
        Names of the GluonTS datasets used for testing.
    num_queries: int
        Number of queries used for the same support set.
    support_set_size: int
        Number of time series in the support set.
    prediction_length: Optional[int]
        Number of predicted time steps.
    context_length_multiple: int
        Context length is prediction length x ``context_length_multiple``.
    support_length_multiple: int
        Support time series length is prediction length x ``support_length_multiple``.
    dataset_sampling: str
         To generate training samples, first a dataset is chosen from the pool of training datasets.
         Then a sample is chosen from the chosen dataset. Training datasets can either be sampled uniformly
         or weighted w.r.t. to the total number of observations.
    data_dir: Path
        Path to data.
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
        dataset_names_train: str,
        dataset_names_val: str,
        dataset_names_test: str,
        num_queries: int,
        support_set_size: int,
        prediction_length: Optional[int] = None,
        context_length_multiple: int = 4,
        support_length_multiple: int = 4,
        dataset_sampling: str = "weighted",
        data_dir: Path = Path.home() / ".mxnet" / "gluon-ts",
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
        self.dataset_names_train = dataset_names_train.split(",")
        self.dataset_names_val = dataset_names_val.split(",")
        self.dataset_names_test = dataset_names_test.split(",")
        self.standardize = standardize

        self.dm_config = {
            "support_set_size": support_set_size,
            "prediction_length": prediction_length,
            "context_length_multiple": context_length_multiple,
            "support_length_multiple": support_length_multiple,
            "standardize": standardize,
            "data_dir": data_dir,
            "batch_size_val_test": batch_size_val_test,
            "num_workers": num_workers,
            "seed": seed,
            "catch22_train": catch22_train,
            "catch22_val_test": catch22_val_test,
        }
        self.data_modules_train = [
            get_data_module(
                name="dm_" + d_name,
                dataset_name=d_name,
                num_queries=num_queries,
                cheat=cheat,
                **self.dm_config,
            )
            for d_name in self.dataset_names_train
        ]
        self.data_modules_val = [
            get_data_module(
                name="dm_" + d_name,
                dataset_name=d_name,
                num_queries=1,
                cheat=float(cheat > 0),
                **self.dm_config,
            )
            for d_name in self.dataset_names_val
        ]
        self.data_modules_test = [
            get_data_module(
                name="dm_" + d_name,
                dataset_name=d_name,
                num_queries=1,
                cheat=float(cheat > 0),
                **self.dm_config,
            )
            for d_name in self.dataset_names_test
        ]
        self.batch_size_train = batch_size_train
        self.batch_size_val_test = batch_size_val_test
        self.dataset_sampling = dataset_sampling
        self.catch22_train = catch22_train
        self.catch22_val_test = catch22_val_test
        self.num_workers = num_workers

    @property
    def prediction_length(self) -> int:
        all_dms = (
            self.data_modules_train
            + self.data_modules_val
            + self.data_modules_test
        )
        return max([dm.prediction_length for dm in all_dms])

    def setup(self, stage: Optional[str] = None) -> None:
        for dm in tqdm(
            self.data_modules_train
            + self.data_modules_val
            + self.data_modules_test,
            desc="setup data loader",
        ):
            dm.setup()

    def train_dataloader(self) -> DataLoader[TripletBatch]:
        super_sampling_dataset = SuperSamplingTripletDataset(
            datasets=[
                dm.sampling_triplet_dataset("train")
                for dm in self.data_modules_train
            ],
            dataset_sampling=self.dataset_sampling,
        )
        return DataLoader(
            super_sampling_dataset,
            collate_fn=TripletBatch.collate,  # type: ignore
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[TripletBatch]:
        return list(
            [dm_val.val_dataloader() for dm_val in self.data_modules_val]
        )

    def test_dataloader(self) -> DataLoader[TripletBatch]:
        return list(
            [dm_test.test_dataloader() for dm_test in self.data_modules_test]
        )

    def get_log_batches(self, n_logging_samples: int) -> Tuple[TripletBatch]:
        its_train = cycle(
            list(
                [
                    iter(dm.sampling_triplet_dataset("train"))
                    for dm in self.data_modules_train
                ]
            )
        )
        its_val = cycle(
            list(
                [
                    iter(dm.sequential_triplet_dataset("val"))
                    for dm in self.data_modules_val
                ]
            )
        )

        def get_log_batch(its):
            samples = []
            for i in range(n_logging_samples):
                samples.append(next(next(its)))
            samples = sorted(
                samples, key=lambda triplet: triplet.query_past[0].dataset_name
            )
            return TripletBatch.collate(samples)

        return get_log_batch(its_train), get_log_batch(its_val)

    @classmethod
    def name(cls) -> str:
        return "dm_super"
