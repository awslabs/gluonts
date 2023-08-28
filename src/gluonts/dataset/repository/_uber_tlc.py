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
import zipfile
from pathlib import Path
from urllib import request

import pandas as pd
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets
from gluonts.dataset.repository._util import metadata


def generate_uber_dataset(
    dataset_path: Path,
    uber_freq: str,
    prediction_length: int,
    dataset_writer: DatasetWriter,
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

    train_data = []
    test_data = []
    for locationID, df in time_series_of_locations:
        df.sort_index()
        df.index = pd.to_datetime(df.index)

        count_series = df.resample(rule=freq_setting).size()
        start_time = pd.Timestamp(df.index[0]).strftime("%Y-%m-%d %X")
        target = count_series.values.tolist()
        feat_static_cat = [locationID]

        test_format_dict = {
            "start": start_time,
            "target": target,
            "feat_static_cat": feat_static_cat,
            "item_id": locationID,
        }
        test_data.append(test_format_dict)

        train_format_dict = {
            "start": start_time,
            "target": target[:-prediction_length],
            "feat_static_cat": feat_static_cat,
            "item_id": locationID,
        }
        train_data.append(train_format_dict)

    meta = MetaData(
        **metadata(
            cardinality=len(time_series_of_locations),
            freq=freq_setting[1],
            prediction_length=prediction_length,
        )
    )

    dataset = TrainDatasets(metadata=meta, train=train_data, test=test_data)
    dataset.save(
        path_str=str(dataset_path), writer=dataset_writer, overwrite=True
    )
