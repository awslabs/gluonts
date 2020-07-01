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

# Standard library imports
import json
import tempfile
from pathlib import Path
from typing import Any, Iterator

# Third-party imports
import numpy as np
import pandas as pd
import pytest
import ujson
from pandas import Timestamp

# First-party imports
from gluonts.dataset.common import (
    FileDataset,
    ListDataset,
    MetaData,
    serialize_data_entry,
)
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.artificial import ComplexSeasonalTimeSeries
from gluonts.dataset.jsonl import JsonLinesFile
from gluonts.dataset.util import find_files
from gluonts.support.util import Timer


def baseline(path: Path, freq: str) -> Iterator[Any]:
    for file in find_files(path, FileDataset.is_valid):
        for line in open(file):
            yield line


def load_json(path: Path, freq: str) -> Iterator[Any]:
    for file in find_files(path, FileDataset.is_valid):
        for line in open(file):
            yield json.loads(line)


def load_ujson(path: Path, freq: str) -> Iterator[Any]:
    for file in find_files(path, FileDataset.is_valid):
        for line in open(file):
            yield ujson.loads(line)


def load_json_lines_file(path: Path, freq: str) -> Iterator[Any]:
    for file in find_files(path, FileDataset.is_valid):
        yield from JsonLinesFile(file)


def load_file_dataset(path: Path, freq: str) -> Iterator[Any]:
    return iter(FileDataset(path, freq))


def load_file_dataset_cached(path: Path, freq: str) -> Iterator[Any]:
    return iter(FileDataset(path, freq, cache=True))


def load_file_dataset_numpy(path: Path, freq: str) -> Iterator[Any]:
    for item in FileDataset(path, freq):
        item["start"] = pd.Timestamp(item["start"])
        item["target"] = np.array(item["target"])
        yield item


def load_parsed_dataset(path: Path, freq: str) -> Iterator[Any]:
    yield from FileDataset(path, freq)


def load_list_dataset(path: Path, freq: str) -> Iterator[Any]:
    lines = (line.content for line in load_json_lines_file(path, freq))
    return iter(ListDataset(lines, freq))


def test_loader_multivariate() -> None:
    with tempfile.TemporaryDirectory() as tmp_folder:
        tmp_path = Path(tmp_folder)

        lines = [
            """{"start": "2014-09-07", "target": [[1, 2, 3]]}
                {"start": "2014-09-07", "target": [[-1, -2, 3], [2, 4, 81]]}
            """,
        ]
        with open(tmp_path / "dataset.json", "w") as f:
            f.write("\n".join(lines))

        ds = list(FileDataset(tmp_path, freq="1D", one_dim_target=False))

        assert (ds[0]["target"] == [[1, 2, 3]]).all()
        assert ds[0]["start"] == Timestamp("2014-09-07", freq="D")

        assert (ds[1]["target"] == [[-1, -2, 3], [2, 4, 81]]).all()
        assert ds[1]["start"] == Timestamp("2014-09-07", freq="D")
