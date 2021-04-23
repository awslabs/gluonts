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

import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp

from gluonts.dataset.artificial import ComplexSeasonalTimeSeries

from gluonts import json
from gluonts.dataset.common import (
    FileDataset,
    ListDataset,
    MetaData,
    ProcessDataEntry,
    serialize_data_entry,
)
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


# @pytest.mark.skip()
def test_io_speed() -> None:
    exp_size = 250
    act_size = 0

    with Timer() as timer:
        dataset = ComplexSeasonalTimeSeries(
            num_series=exp_size,
            freq_str="D",
            length_low=100,
            length_high=200,
            min_val=0.0,
            max_val=20000.0,
            proportion_missing_values=0.1,
        ).generate()
    print(f"Test data generation took {timer.interval} seconds")

    # name of method, loading function and min allowed throughput
    fixtures = [
        ("baseline", baseline, 60_000),
        # ('json.loads', load_json, xxx),
        ("json.loads", load_json, 20_000),
        ("JsonLinesFile", load_json_lines_file, 10_000),
        ("ListDataset", load_list_dataset, 500),
        ("FileDataset", load_file_dataset, 500),
        ("FileDatasetCached", load_file_dataset_cached, 500),
        ("FileDatasetNumpy", load_file_dataset_numpy, 500),
        ("ParsedDataset", load_parsed_dataset, 500),
    ]

    with tempfile.TemporaryDirectory() as path:
        # save the generated dataset to a temporary folder
        with Timer() as timer:
            dataset.save(path)
        print(f"Test data saving took {timer.interval} seconds")

        # for each loader, read the dataset, assert that the number of lines is
        # correct, and record the lines/sec rate
        rates = {}
        for name, get_loader, _ in fixtures:
            with Timer() as timer:
                loader = get_loader(
                    Path(path) / "train", dataset.metadata.freq
                )
                for act_size, _ in enumerate(loader, start=1):
                    pass
            rates[name] = int(act_size / max(timer.interval, 0.00001))
            print(
                f"Loader {name:13} achieved a rate of {rates[name]:10} "
                f"lines/second"
            )
            assert exp_size == act_size, (
                f"Loader {name} did not yield the expected number of "
                f"{exp_size} lines"
            )

        # for each loader, assert that throughput is above threshold
        for name, _, min_rate in fixtures:
            assert min_rate <= rates[name], (
                f"The throughput of {name} ({rates[name]} lines/second) "
                f"was below the allowed maximum rate {min_rate}."
            )


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
