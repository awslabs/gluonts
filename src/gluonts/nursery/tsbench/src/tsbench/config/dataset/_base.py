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

from __future__ import annotations
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, Dict, Tuple, Union
import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset, FileDataset, MetaData
from numpy import ma
from tsbench.constants import (
    DEFAULT_DATA_CATCH22_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DATA_STATS_PATH,
)


@dataclass(frozen=True)
class DatasetConfig:
    """
    A dataset configuration references a dataset containing multiple time
    series.
    """

    base_path: Path = field(default=DEFAULT_DATA_PATH, compare=False)

    @classmethod
    def name(cls) -> str:
        """
        Returns a canonical name for the dataset.
        """
        raise NotImplementedError

    def generate(self) -> None:
        """
        Downloads the dataset into the globally configured data directory and
        applies necessary preprocessing steps.

        This function must be called on a machine prior to using the dataset.
        """
        raise NotImplementedError

    def prepare(self) -> None:
        """
        Generates all necessary representations of the dataset after it has
        been generated.
        """
        self.data.train().prepare()
        self.data.val().prepare()
        self.data.test().prepare()

    @property
    def has_time_features(self) -> bool:
        """
        Returns whether the dataset has time features.
        """
        return True

    @property
    def max_training_time(self) -> int:
        """
        Returns the maximum training time in seconds for the dataset.
        """
        raise NotImplementedError

    @property
    def meta(self) -> MetaData:
        """
        Returns the dataset's metadata.
        """
        return MetaData.parse_file(self.root / "gluonts" / "metadata.json")

    @property
    def data(self) -> DatasetSplits:
        """
        Returns the dataset's splits, i.e. training, validation, and test data.

        This is a noop, the datasets are only loaded at a later point.
        """
        return DatasetSplits(self.meta, self.root)

    @property
    def root(self) -> Path:
        """
        Returns the directory where all the data pertaining to this dataset is
        stored.
        """
        return self.base_path / self.name()

    def stats(
        self, root: str | Path = DEFAULT_DATA_STATS_PATH
    ) -> dict[str, float]:
        """
        Returns basic statistics of the dataset.

        Args:
            root: The directory which contains the stats files.
        """
        file = Path(root) / f"{self.name()}.json"
        with file.open("r") as f:
            return json.load(f)

    def catch22(
        self, root: str | Path = DEFAULT_DATA_CATCH22_PATH
    ) -> pd.DataFrame:
        """
        Returns the catch22 features of all time series in the dataset.

        Args:
            root: The directory which contains the catch22 feature files.
        """
        file = Path(root) / f"{self.name()}.parquet"
        return pd.read_parquet(file)


@dataclass
class DatasetSplits:
    """
    The dataset splits provide train, validation and test data for a particular
    dataset.

    Calling any of the functions here, is a noop. Data is only loaded once a
    particular representation of the data is accessed.
    """

    _metadata: MetaData
    _directory: Path

    def train(self, val: bool = True) -> DatasetSplit:
        """
        Returns the train data for the dataset.

        Args:
            val: Whether validation data is used. If not, this returns the validation data, i.e.
                the same time series that are longer by the prediction length.
        """
        return DatasetSplit(
            self._metadata, self._directory, "train" if val else "val"
        )

    def val(self) -> DatasetSplit:
        """
        Returns the validation data for the dataset.

        This is the same as :meth:`train(False)`.
        """
        return DatasetSplit(self._metadata, self._directory, "val")

    def test(self) -> DatasetSplit:
        """
        Returns the test data for the dataset.
        """
        return DatasetSplit(self._metadata, self._directory, "test")


@dataclass
class DatasetSplit:
    """
    A dataset split provides all the representations for a particular split
    (train/val/test) of a dataset.
    """

    _metadata: MetaData
    _directory: Path
    _split: str

    def gluonts(self) -> Dataset:
        """
        Returns the GluonTS dataset for the dataset split.

        This loads the associated JSON file and is, thus, potentially slow.
        """
        return FileDataset(
            self._directory / "gluonts" / self._split, freq=self._metadata.freq
        )

    def evaluation(self) -> EvaluationDataset:
        """
        Returns the NumPy arrays that are used to perform evaluation.
        """
        if self._split == "train":
            raise ValueError(
                "training data does not provide an evaluation dataset"
            )

        base = self._directory / "numpy" / self._split
        return EvaluationDataset(
            np.load(base / "future_data.npy"),
            ma.MaskedArray(
                np.load(base / "past_data.npy"),
                mask=np.load(base / "past_mask.npy"),
            ),
        )

    def prepare(self) -> None:
        """
        Prepares all required representations provided that the GluonTS dataset
        is already generated.
        """
        target = self._directory / "numpy" / self._split
        if self._split == "train":
            if target.exists():
                shutil.rmtree(target)
            return

        if target.exists():
            if (
                (target / "future_data.npy").exists()
                and (target / "past_data.npy").exists()
                and (target / "past_mask.npy").exists()
            ):
                return
            shutil.rmtree(target)

        target.mkdir(parents=True)
        future, past = _generate_evaluation_dataset(
            self.gluonts(), cast(int, self._metadata.prediction_length)
        )

        np.save(target / "future_data.npy", future)
        np.save(target / "past_data.npy", past.data)
        np.save(target / "past_mask.npy", past.mask)


@dataclass
class EvaluationDataset:
    """
    The evaluation dataset is a simple dataset representation that contains a
    two-dimensional array of future values as well as a two-dimensional
    (masked) array of the past values that a model sees during training.

    This representation is very efficient for evaluation.
    """

    future: np.ndarray
    past: ma.MaskedArray


# -------------------------------------------------------------------------------------------------


def _generate_evaluation_dataset(
    dataset: Dataset, prediction_length: int
) -> tuple[np.ndarray, ma.MaskedArray]:
    # Extract data from all the values in the dataset
    pasts = []
    past_lengths = []
    predictions = []
    for item in dataset:
        target = item["target"]
        pasts.append(target[:-prediction_length])
        past_lengths.append(target.shape[0] - prediction_length)
        predictions.append(target[-prediction_length:])

    # Compute masked past values as well as (non-masked) future values
    max_len = np.max(past_lengths)
    values = np.empty((len(dataset), max_len))
    mask = np.ones((len(dataset), max_len), dtype=bool)
    for i in range(len(dataset)):
        values[i, : past_lengths[i]] = pasts[i]
        mask[i, : past_lengths[i]] = False

    y_past = ma.masked_array(values, mask=mask)
    y_true = np.stack(predictions)
    return y_true, y_past
