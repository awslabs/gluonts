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

import pytest
from gluonts.dataset.repository.datasets import dataset_names, get_dataset


def check_train_test_split(dataset):
    prediction_length = dataset.metadata.prediction_length

    train_end = {}
    for entry in dataset.train:
        assert (
            entry["item_id"] not in train_end
        ), f"item {entry['item_id']} is duplicate"
        train_end[entry["item_id"]] = (
            entry["start"] + len(entry["target"]) * entry["start"].freq
        )

    test_end = {}
    for entry in dataset.test:
        test_end[entry["item_id"]] = (
            entry["start"] + len(entry["target"]) * entry["start"].freq
        )

    for k in test_end:
        if k not in train_end:
            continue
        expected_end = train_end[k] + prediction_length * train_end[k].freq
        assert (
            test_end[k] >= expected_end
        ), f"test entry for item {k} ends at {test_end[k]} < {expected_end}"


@pytest.mark.skip
@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", dataset_names)
def test_data_leakage(name):
    try:
        dataset = get_dataset(name)
    except RuntimeError:
        print(f"WARN dataset '{name}' could not be obtained")

    check_train_test_split(dataset)
