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
import tempfile
import zipfile
from pathlib import Path
from urllib import request

import pandas as pd

from gluonts.dataset.repository._util import metadata


def generate_uber_dataset(
    dataset_path: Path, uber_freq: str, prediction_length: int
):
    subsets = {"daily": "1D", "hourly": "1H"}
    assert (
        uber_freq.lower() in subsets
    ), f"invalid uber_freq='{uber_freq}'. Allowed values: {subsets.keys()}"
    freq_setting = subsets[uber_freq.lower()]

    # download the dataset and read the data
    with tempfile.TemporaryDirectory() as dir_path:
        temp_dir_path = Path(dir_path)
        temp_zip_path = temp_dir_path / "uber-dataset.zip"
        uber_url_path = (
            "http://raw.githubusercontent.com/fivethirtyeight/"
            "uber-tlc-foil-response/master/uber-trip-data/"
            "uber-raw-data-janjune-15.csv.zip"
        )
        request.urlretrieve(uber_url_path, temp_zip_path)
        with zipfile.ZipFile(temp_zip_path) as zf:
            zf.extractall(path=temp_dir_path)
        uber_file_path = temp_dir_path / "uber-raw-data-janjune-15.csv"
        uber_df = pd.read_csv(
            uber_file_path,
            header=0,
            usecols=["Pickup_date", "locationID"],
            index_col=0,
        )

    # We divide the raw data according to locationID. Each json line represents
    # a time series of a loacationID. The targets are numbers of pickup-events
    # during a day or an hour.
    time_series_of_locations = list(uber_df.groupby(by="locationID"))

    dataset_path.mkdir(exist_ok=True)
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"
    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)

    train_file = train_path / "data.json"
    test_file = test_path / "data.json"
    with open(train_file, "w") as o_train, open(test_file, "w") as o_test:
        for locationID, df in time_series_of_locations:
            df.sort_index()
            df.index = pd.to_datetime(df.index)

            count_series = df.resample(rule=freq_setting).size()
            start_time = pd.Timestamp(df.index[0]).strftime("%Y-%m-%d %X")
            target = count_series.values.tolist()
            feat_static_cat = [locationID]
            format_dict = {
                "start": start_time,
                "target": target,
                "feat_static_cat": feat_static_cat,
            }
            test_json_line = json.dumps(format_dict)
            o_test.write(test_json_line)
            o_test.write("\n")
            format_dict["target"] = format_dict["target"][:-prediction_length]
            train_json_line = json.dumps(format_dict)
            o_train.write(train_json_line)
            o_train.write("\n")

    with open(dataset_path / "metadata.json", "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=len(time_series_of_locations),
                    freq=freq_setting[1],
                    prediction_length=prediction_length,
                )
            )
        )
