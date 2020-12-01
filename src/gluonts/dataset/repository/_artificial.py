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
from pathlib import Path

from gluonts.dataset.artificial import ArtificialDataset
from gluonts.dataset.artificial.generate_synthetic import generate_sf2
from gluonts.dataset.common import serialize_data_entry


def generate_artificial_dataset(
    dataset_path: Path, dataset: ArtificialDataset
) -> None:
    dataset_path_train = dataset_path / "train"
    dataset_path_test = dataset_path / "test"

    dataset_path.mkdir(exist_ok=True)
    dataset_path_train.mkdir(exist_ok=False)
    dataset_path_test.mkdir(exist_ok=False)

    ds = dataset.generate()
    assert ds.test is not None

    with (dataset_path / "metadata.json").open("w") as fp:
        json.dump(ds.metadata.dict(), fp, indent=2, sort_keys=True)

    generate_sf2(
        filename=str(dataset_path_train / "train.json"),
        time_series=list(map(serialize_data_entry, ds.train)),
        is_missing=False,
        num_missing=0,
    )

    generate_sf2(
        filename=str(dataset_path_test / "test.json"),
        time_series=list(map(serialize_data_entry, ds.test)),
        is_missing=False,
        num_missing=0,
    )
