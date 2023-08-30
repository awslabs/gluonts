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
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
from gluonts.dataset.common import FileDataset, MetaData
from gluonts.dataset.field_names import FieldName
import torch
import numpy as np
import numpy.ma as ma
import shutil

from meta.data.dataset import TimeSeries, TimeSeriesDataset
from meta.common import tensor_to_np


@dataclass
class DatasetSplits:
    """
    The dataset splits provide train, validation and test data for a particular dataset. Calling
    any of the functions here, is a noop. Data is only loaded once a particular representation of
    the data is accessed.
    """

    _metadata: MetaData
    _directory: Path
    _dataset_name: str
    _prediction_length: int
    _standardize: bool

    def train(self, val: bool = True, name: str = "") -> DatasetSplit:
        """
        Returns the train data for the dataset.

        Args:
            val: Whether validation data is used. If not, this returns the validation data, i.e.
                the same time series that are longer by the prediction length.
            name: A specific dataset name. Should only be used for the cheat data module.
        """
        if val:
            split_name = f"train_{name}" if name else "train"
        else:
            split_name = f"val_{name}" if name else "val"

        return DatasetSplit(
            self._metadata,
            self._directory,
            split_name,
            self._dataset_name,
            _standardize=self._standardize,
        )

    def val(self, name: str = "") -> DatasetSplit:
        """
        Returns the validation data for the dataset. This is the same as :meth:`train(False)`.
        """
        return DatasetSplit(
            self._metadata,
            self._directory,
            f"val_{name}" if name else "val",
            self._dataset_name,
            self._standardize,
            self._prediction_length,
        )

    def test(self, name: str = "") -> DatasetSplit:
        """
        Returns the test data for the dataset.
        """
        return DatasetSplit(
            self._metadata,
            self._directory,
            f"test_{name}" if name else "test",
            self._dataset_name,
            self._standardize,
            self._prediction_length,
        )


@dataclass
class DatasetSplit:
    """
    A dataset split provides all the representations for a particular split (train/val/test) of a
    dataset.
    """

    _metadata: MetaData
    _directory: Path
    _split: str
    _dataset_name: str
    _standardize: bool
    _prediction_length: Optional[int] = None

    def data(self, evaluation: bool = False) -> TimeSeriesDataset:
        """
        Returns a time series dataset for the dataset split. This loads the associated JSON file and
        is, thus, potentially slow.
        """
        gluonts = FileDataset(
            self._directory / self._split, freq=self._metadata.freq
        )
        series = [
            TimeSeries(
                dataset_name=self._dataset_name,
                values=torch.from_numpy(item[FieldName.TARGET]).unsqueeze(-1),
                item_id=item[FieldName.ITEM_ID],
                start_date=item[FieldName.START],
            )
            for item in gluonts
        ]
        dataset = TimeSeriesDataset(
            series=series,
            prediction_length=self._metadata.prediction_length,
            freq=self._metadata.freq,
            # do not standardize evaluation dataset
            standardize=self._standardize and not evaluation,
        )
        return dataset

    def support_set(self) -> List[List[TimeSeries]]:
        support_set_path = self._directory / self._split / ".support_set.json"
        assert (
            support_set_path.exists()
        ), "No support set has been saved for this dataset."
        support_set = []
        with open(support_set_path) as json_file:
            for line in json_file:
                series = [
                    TimeSeries(
                        dataset_name=self._dataset_name,
                        values=torch.FloatTensor(s["target"]).unsqueeze(-1),
                        item_id=s["item_id"],
                        start_date=s["start"],
                    )
                    for s in json.loads(line)
                ]
                support_set.append(series)
        return support_set

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
            np.load(base / "future_data.npy")[..., : self._prediction_length],
            ma.MaskedArray(
                np.load(base / "past_data.npy"),
                mask=np.load(base / "past_mask.npy"),
            ),
        )

    def prepare(self) -> None:
        """
        Prepares all required representations provided that the GluonTS dataset is already
        generated.
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
        # Evaluation dataset is not standardized
        future, past = _generate_evaluation_dataset(
            self.data(evaluation=True), self._metadata.prediction_length
        )

        np.save(target / "future_data.npy", future)
        np.save(target / "past_data.npy", past.data)
        np.save(target / "past_mask.npy", past.mask)


@dataclass
class EvaluationDataset:
    """
    The evaluation dataset is a simple dataset representation that contains a two-dimensional array
    of future values as well as a two-dimensional (masked) array of the past values that a model
    sees during training. This representation is very efficient for evaluation.
    """

    future: np.ndarray
    past: ma.MaskedArray


# -------------------------------------------------------------------------------------------------


def _generate_evaluation_dataset(
    dataset: TimeSeriesDataset, prediction_length: int
) -> Tuple[np.ndarray, ma.MaskedArray]:
    # Extract data from all the values in the dataset
    pasts = []
    past_lengths = []
    predictions = []
    for item in dataset:
        target = tensor_to_np(item.values).squeeze()
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
