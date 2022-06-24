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
from this import d
from typing import List, Optional

from gluonts.dataset import DatasetWriter
from gluonts.dataset.artificial import ArtificialDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._util import create_dataset_paths
from gluonts.dataset.jsonl import encode_json


def generate_artificial_dataset(
    dataset_path: Path,
    dataset: ArtificialDataset,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
) -> None:
    paths = create_dataset_paths(dataset_path, ["train", "test"])

    ds = dataset.generate()
    assert ds.test is not None
    if prediction_length is not None:
        ds.metadata.prediction_length = prediction_length

    with (dataset_path / "metadata.json").open("w") as fp:
        json.dump(ds.metadata.dict(), fp, indent=2, sort_keys=True)

    generate_sf2(
        path=paths["train"],
        time_series=list(map(encode_json, ds.train)),
        is_missing=False,
        dataset_writer=dataset_writer,
        num_missing=0,
    )

    generate_sf2(
        path=paths["test"],
        time_series=list(map(encode_json, ds.test)),
        is_missing=False,
        dataset_writer=dataset_writer,
        num_missing=0,
    )


def generate_sf2(
    path: Path,
    time_series: List,
    dataset_writer: DatasetWriter,
    is_missing: bool,
    num_missing: int,
) -> None:
    data = []
    for ts in time_series:
        if is_missing:
            target = []  # type: List
            # For Forecast don't output feat_static_cat and
            # feat_static_real
            for j, val in enumerate(ts[FieldName.TARGET]):
                # only add ones that are not missing
                if j != 0 and j % num_missing == 0:
                    target.append(None)
                else:
                    target.append(val)
            ts[FieldName.TARGET] = target
        ts.pop(FieldName.FEAT_STATIC_CAT, None)
        ts.pop(FieldName.FEAT_STATIC_REAL, None)
        # Chop features in training set
        if FieldName.FEAT_DYNAMIC_REAL in ts.keys() and "train" in str(path):
            # TODO: Fix for missing values
            for i, feat_dynamic_real in enumerate(
                ts[FieldName.FEAT_DYNAMIC_REAL]
            ):
                ts[FieldName.FEAT_DYNAMIC_REAL][i] = feat_dynamic_real[
                    : len(ts[FieldName.TARGET])
                ]
        data.append(ts)
    dataset_writer.write_to_folder(data, path)
