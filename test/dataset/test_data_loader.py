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

import json
import sys
from collections import Counter
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any, List

import numpy as np
import pytest

from gluonts.core.component import equals
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.common import DataEntry, DataBatch, FileDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    Batch,
    TrainDataLoader,
    ValidationDataLoader,
    InferenceDataLoader,
)
from gluonts.transform import (
    InstanceSampler,
    InstanceSplitter,
    MapTransformation,
)

from gluonts.testutil.batchify import batchify


class WriteIsTrain(MapTransformation):
    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data["is_train"] = is_train
        return data


class ExactlyOneSampler(InstanceSampler):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        assert window_size > 0
        return np.array([a])


@contextmanager
def default_list_dataset():
    yield constant_dataset()[1]


@contextmanager
def default_file_dataset():
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        with open(temp_dir / "file.json", "w") as fp:
            for item_id in range(10):
                json.dump(
                    {
                        "item_id": item_id,
                        "start": "2021-01-01 00:00:00",
                        "target": [1.0, 2.0, 3.0, 4.0, 5.0],
                    },
                    fp,
                )
                fp.write("\n")

        yield FileDataset(temp_dir, "H")


def default_transformation():
    return WriteIsTrain() + InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExactlyOneSampler(),
        past_length=10,
        future_length=5,
        dummy_value=1.0,
    )


def count_item_ids(batches: List[DataBatch]) -> Dict[Any, int]:
    return Counter(
        item_id for batch in batches for item_id in batch[FieldName.ITEM_ID]
    )


@pytest.mark.parametrize(
    "dataset_context",
    (
        [
            default_list_dataset,
            default_file_dataset,
        ]
        if not sys.platform.startswith("win")
        else [default_list_dataset]
    ),
)
def test_training_data_loader(dataset_context):
    with dataset_context() as dataset:
        dataset_length = len(dataset)

        batch_size = 4

        dl = TrainDataLoader(
            dataset=dataset,
            transform=default_transformation(),
            batch_size=batch_size,
            stack_fn=batchify,
        )

        num_epochs = 20
        epoch_length = 2

        passes_through_dataset = int(
            (num_epochs * epoch_length * batch_size) / dataset_length
        )

        # these are to make sure that the test makes sense:
        # we want to go over the dataset multiple times
        assert passes_through_dataset >= 10
        # we want each epoch to be shorter than the dataset
        assert epoch_length * batch_size < dataset_length

        batches = []

        for epoch in range(num_epochs):
            for batch in islice(dl, epoch_length):
                assert all(batch["is_train"])
                batches.append(batch)

        counter = count_item_ids(batches)

        for entry in dataset:
            assert counter[entry[FieldName.ITEM_ID]] >= 1


@pytest.mark.parametrize(
    "dataset_context",
    [
        default_list_dataset,
        default_file_dataset,
    ],
)
def test_validation_data_loader(dataset_context):
    with dataset_context() as dataset:
        dl = ValidationDataLoader(
            dataset=dataset,
            transform=default_transformation(),
            batch_size=4,
            stack_fn=batchify,
        )

        for _ in range(3):
            batches = list(dl)

            for batch in batches:
                assert all(batch["is_train"])

            counter = count_item_ids(batches)

            for entry in dataset:
                assert counter[entry[FieldName.ITEM_ID]] == 1


@pytest.mark.parametrize(
    "dataset_context",
    [
        default_list_dataset,
        default_file_dataset,
    ],
)
def test_inference_data_loader(dataset_context):
    with dataset_context() as dataset:
        dl = InferenceDataLoader(
            dataset=dataset,
            transform=default_transformation(),
            batch_size=4,
            stack_fn=batchify,
        )

        batches = list(dl)

        for batch in batches:
            assert not any(batch["is_train"])

        counter = count_item_ids(batches)

        for entry in dataset:
            assert counter[entry[FieldName.ITEM_ID]] == 1


def test_equals_batch():
    assert equals(Batch(batch_size=10), Batch(batch_size=10))
    assert not equals(Batch(batch_size=10), Batch(batch_size=100))
