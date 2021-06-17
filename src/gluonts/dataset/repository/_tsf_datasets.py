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

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, NamedTuple
from urllib import request
from zipfile import ZipFile

from gluonts import json
from gluonts.dataset import jsonl
from gluonts.gluonts_tqdm import tqdm

from ._tsf_reader import frequency_converter, TSFReader
from ._util import metadata, to_dict, request_retrieve_hook


class Dataset(NamedTuple):
    file_name: str
    record: str
    ROOT: str = "https://zenodo.org/record"

    @property
    def url(self):
        return f"{self.ROOT}/{self.record}/files/{self.file_name}"

    def download(self, path: Path):
        file_path = path / self.file_name
        with tqdm(
            [],
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=5,
            desc=f"Download {self.file_name}:",
        ) as _tqdm:
            request.urlretrieve(
                self.url,
                filename=file_path,
                reporthook=request_retrieve_hook(_tqdm),
            )
        return file_path


datasets = {
    "kaggle_web_traffic_with_missing": Dataset(
        file_name="kaggle_web_traffic_dataset_with_missing_values.zip",
        record="4656080",
    ),
    "kaggle_web_traffic_without_missing": Dataset(
        file_name="kaggle_web_traffic_dataset_without_missing_values.zip",
        record="4656075",
    ),
    "kaggle_web_traffic_weekly": Dataset(
        file_name="kaggle_web_traffic_weekly_dataset.zip",
        record="4656664",
    ),
    "m1_yearly": Dataset(file_name="m1_yearly_dataset.zip", record="4656193"),
    "m1_quarterly": Dataset(
        file_name="m1_quarterly_dataset.zip", record="4656154"
    ),
    "m1_monthly": Dataset(
        file_name="m1_monthly_dataset.zip", record="4656159"
    ),
    "nn5_daily_with_missing": Dataset(
        file_name="nn5_daily_dataset_with_missing_values.zip",
        record="4656110",
    ),
    "nn5_daily_without_missing": Dataset(
        file_name="nn5_daily_dataset_without_missing_values.zip",
        record="4656117",
    ),
    "nn5_weekly": Dataset(
        file_name="nn5_weekly_dataset.zip", record="4656125"
    ),
    "tourism_monthly": Dataset(
        file_name="tourism_monthly_dataset.zip",
        record="4656096",
    ),
    "tourism_quarterly": Dataset(
        file_name="tourism_quarterly_dataset.zip",
        record="4656093",
    ),
    "tourism_yearly": Dataset(
        file_name="tourism_yearly_dataset.zip",
        record="4656103",
    ),
}


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

            jsonl.dump([dic], test_fp)

            dic["target"] = dic["target"][:-train_offset]
            jsonl.dump([dic], train_fp)


def generate_forecasting_dataset(dataset_path: Path, dataset_name: str):
    dataset = datasets[dataset_name]
    dataset_path.mkdir(exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        with ZipFile(dataset.download(temp_path)) as archive:
            archive.extractall(path=temp_path)

        # only one file is exptected
        reader = TSFReader(temp_path / archive.namelist()[0])
        meta, data = reader.read()

    prediction_length = int(meta.forecast_horizon)

    save_metadata(
        dataset_path,
        len(data),
        frequency_converter(meta.frequency),
        prediction_length,
    )

    save_datasets(dataset_path, data, prediction_length)
