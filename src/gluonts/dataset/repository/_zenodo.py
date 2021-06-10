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

import os
from pathlib import Path
from typing import List, Dict
from urllib import request
from zipfile import ZipFile

from ._tsf_reader import TSFReader
from ._util import metadata, to_dict, save_to_file

from gluonts import json
from gluonts.dataset.jsonl import dump
from gluonts.gluonts_tqdm import tqdm

ROOT = "https://zenodo.org/record"

dataset_info = {
    "kaggle_web_traffic_with_missing": {
        "file": "kaggle_web_traffic_dataset_with_missing_values.zip",
        "record": "4656080",
    },
    "kaggle_web_traffic_without_missing": {
        "file": "kaggle_web_traffic_dataset_without_missing_values.zip",
        "record": "4656075",
    },
    "kaggle_web_traffic_weekly": {
        "file": "kaggle_web_traffic_weekly_dataset.zip",
        "record": "4656664",
    },
    "m1_yearly": {"file": "m1_yearly_dataset.zip", "record": "4656193"},
    "m1_quarterly": {"file": "m1_quarterly_dataset.zip", "record": "4656154"},
    "m1_monthly": {"file": "m1_monthly_dataset.zip", "record": "4656159"},
    "nn5_daily_with_missing": {
        "file": "nn5_daily_dataset_with_missing_values.zip",
        "record": "4656110",
    },
    "nn5_daily_without_missing": {
        "file": "nn5_daily_dataset_without_missing_values.zip",
        "record": "4656117",
    },
    "nn5_weekly": {"file": "nn5_weekly_dataset.zip", "record": "4656125"},
    "tourism_monthly": {
        "file": "tourism_monthly_dataset.zip",
        "record": "4656096",
    },
    "tourism_quarterly": {
        "file": "tourism_quarterly_dataset.zip",
        "record": "4656093",
    },
    "tourism_yearly": {
        "file": "tourism_yearly_dataset.zip",
        "record": "4656103",
    },
}


def urllib_retrieve_hook(tqdm):
    """Wraps tqdm instance.
    Don'tqdm forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    # >>> with tqdm(...) as tqdm:
    # ...     reporthook = my_hook(tqdm)
    # ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(block=1, block_size=1, tsize=None):
        """
        block  : int, optional
            Number of blocks transferred so far [default: 1].
        block_size  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            tqdm.total = tsize
        tqdm.update((block - last_b[0]) * block_size)
        last_b[0] = block

    return update_to


def download_dataset(description: Dict[str, str], path: Path):
    file = description["file"]
    with tqdm(
        [],
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=5,
        desc=f"download {file}",
    ) as _tqdm:
        request.urlretrieve(
            f"{ROOT}/{description['record']}/files/{file}",
            filename=str(path / description["file"]),
            reporthook=urllib_retrieve_hook(_tqdm),
        )


def frequency_converter(freq: str):
    parts = freq.split("_")
    if len(parts) == 1:
        return freq[0].upper()
    if len(parts) == 2 and parts[0].isnumeric():
        return f"{parts[0]}{parts[1].upper()}"


def save_metadata(
    dataset_path: Path, cardinality: int, freq: str, prediction_length: int
):
    with open(dataset_path / "metadata.json", "w") as file:
        json.dump(
            metadata(
                cardinality=cardinality,
                freq=freq,
                prediction_length=prediction_length,
            ),
            file,
        )


def save_datasets(path: Path, data: List[Dict], train_offset: int):
    train = path / "train"
    test = path / "test"

    train.mkdir(exist_ok=True)
    test.mkdir(exist_ok=True)

    with open(train / "data.json", "w") as train_fp, open(
        test / "data.json", "w"
    ) as test_fp:
        for data_entry in tqdm(
            data, total=len(data), desc="creating json files"
        ):
            dic = to_dict(
                target_values=data_entry["target"],
                start=str(data_entry["start_timestamp"]),
            )

            test_fp.write(json.dumps(dic))
            test_fp.write("\n")

            dic["target"] = dic["target"][:-train_offset]
            train_fp.write(json.dumps(dic))
            train_fp.write("\n")


def clean_up_dataset(dataset_path: Path, file_names: List[str]):
    for file in file_names:
        file_path = dataset_path / file
        file_path.unlink()


def generate_forecasting_dataset(dataset_path: Path, dataset_name: str):
    ds_info = dataset_info[dataset_name]
    dataset_path.mkdir(exist_ok=True)

    file = ds_info["file"]
    file_path = dataset_path / file
    download_dataset(ds_info, dataset_path)
    with ZipFile(file_path, "r") as zip:
        file_names = zip.namelist()
        # TODO better exception text
        assert len(file_names) == 1, "Too many files"
        for member in tqdm(zip.infolist(), desc=f"Extracting {file}"):
            zip.extract(member, str(dataset_path))
    reader = TSFReader(dataset_path / file_names[0])
    meta, data = reader.read()

    prediction_length = int(meta.forecast_horizon)
    save_metadata(
        dataset_path,
        len(data),
        frequency_converter(meta.frequency),
        prediction_length,
    )
    save_datasets(dataset_path, data, prediction_length)

    file_names.append(file)
    clean_up_dataset(dataset_path, file_names)
