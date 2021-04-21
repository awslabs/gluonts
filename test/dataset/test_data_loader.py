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

from collections import defaultdict
from functools import partial
from itertools import islice
from typing import Dict, Any, List
from tempfile import TemporaryDirectory
from pathlib import Path
from contextlib import AbstractContextManager
import sys
import time

from mxnet.context import current_context
import numpy as np
import pytest

from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.common import DataEntry, DataBatch, FileDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    TrainDataLoader,
    ValidationDataLoader,
    InferenceDataLoader,
)
from gluonts.mx.batchify import batchify, as_in_context
from gluonts.transform import (
    InstanceSampler,
    InstanceSplitter,
    MapTransformation,
    SimpleTransformation,
)


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


class DefaultListDataset(AbstractContextManager):
    def __enter__(self):
        return constant_dataset()[1]

    def __exit__(self, *exec_details):
        pass


class DefaultFileDataset(AbstractContextManager):
    def __enter__(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = self.tempdir.__enter__()
        file_path = Path(self.tempdir_path) / "file.json"
        with open(file_path, "w") as fp:
            for item_id in range(10):
                line = f'"item_id": {item_id}, "start": "2021-01-01 00:00:00", "target": [1.0, 2.0, 3.0, 4.0, 5.0]'
                fp.write("{" + line + "}\n")
        return FileDataset(self.tempdir_path, "H")

    def __exit__(self, *args, **kwargs):
        self.tempdir.__exit__(*args, **kwargs)


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
    counter: Dict[Any, int] = defaultdict(lambda: 0)
    for batch in batches:
        for item_id in batch[FieldName.ITEM_ID]:
            counter[item_id] += 1
    return counter


@pytest.mark.parametrize(
    "dataset_context",
    (
        [
            DefaultListDataset(),
            DefaultFileDataset(),
        ]
        if not sys.platform.startswith("win")
        else [DefaultListDataset()]
    ),
)
@pytest.mark.parametrize(
    "num_workers",
    [None, 1, 2, 5],
)
def test_training_data_loader(dataset_context, num_workers):
    with dataset_context as dataset:
        dataset_length = len(list(dataset))

        batch_size = 4

        dl = TrainDataLoader(
            dataset=dataset,
            transform=default_transformation(),
            batch_size=batch_size,
            stack_fn=partial(batchify, ctx=current_context()),
            decode_fn=partial(as_in_context, ctx=current_context()),
            num_workers=num_workers,
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
                assert all(x is True for x in batch["is_train"])
                batches.append(batch)

        counter = count_item_ids(batches)

        if num_workers is None or num_workers == 1:
            for entry in dataset:
                assert counter[entry[FieldName.ITEM_ID]] >= 1


@pytest.mark.parametrize(
    "dataset_context",
    [
        DefaultListDataset(),
        DefaultFileDataset(),
    ],
)
def test_validation_data_loader(dataset_context):
    with dataset_context as dataset:
        dataset_length = len(list(dataset))
        counter = defaultdict(lambda: 0)

        dl = ValidationDataLoader(
            dataset=dataset,
            transform=default_transformation(),
            batch_size=4,
            stack_fn=partial(batchify, ctx=current_context()),
        )

        batches = list(dl)

        for batch in batches:
            assert all(x is True for x in batch["is_train"])

        counter = count_item_ids(batches)

        for entry in dataset:
            assert counter[entry[FieldName.ITEM_ID]] == 1

        batches_again = list(dl)

        assert (b1 == b2 for b1, b2 in zip(batches, batches_again))


@pytest.mark.parametrize(
    "dataset_context",
    [
        DefaultListDataset(),
        DefaultFileDataset(),
    ],
)
def test_inference_data_loader(dataset_context):
    with dataset_context as dataset:
        dataset_length = len(list(dataset))
        counter = defaultdict(lambda: 0)

        dl = InferenceDataLoader(
            dataset=dataset,
            transform=default_transformation(),
            batch_size=4,
            stack_fn=partial(batchify, ctx=current_context()),
        )

        batches = list(dl)

        for batch in batches:
            assert all(x is False for x in batch["is_train"])

        counter = count_item_ids(batches)

        for entry in dataset:
            assert counter[entry[FieldName.ITEM_ID]] == 1
