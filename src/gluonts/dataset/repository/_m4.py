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

import pandas as pd

from gluonts.dataset import pandas, DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets
from gluonts.dataset.repository._util import metadata


def generate_m4_dataset(
    dataset_path: Path,
    m4_freq: str,
    pandas_freq: str,
    prediction_length: int,
    dataset_writer: DatasetWriter,
):
    m4_dataset_url = (
        "https://github.com/M4Competition/M4-methods/raw/master/Dataset"
    )
    meta_info = pd.read_csv(f"{m4_dataset_url}/M4-info.csv", index_col=0)
    train_df = pd.read_csv(
        f"{m4_dataset_url}/Train/{m4_freq}-train.csv", index_col=0
    )
    test_df = pd.read_csv(
        f"{m4_dataset_url}/Test/{m4_freq}-test.csv", index_col=0
    )

    train_dict = {}
    test_dict = {}
    for idx, row in train_df.iterrows():
        target = row.dropna(axis=0)
        # some time series have more than 300 years which can not be
        # represented in pandas, this is probably due to a misclassification
        # of those time series as Yearly. We use only those time series with
        # fewer than 300 items for this reason.
        if m4_freq == "Yearly" and target.shape[0] > 300:
            continue
        start = pd.date_range(
            meta_info.loc[idx, "StartingDate"],
            freq=pandas_freq,
            periods=target.shape[0],
        )
        train_dict[idx] = pd.DataFrame({"target": target, "start": start})

    for idx, row in test_df.iterrows():
        target = row.dropna(axis=0)
        if m4_freq == "Yearly" and target.shape[0] > 300:
            continue
        start = pd.date_range(
            meta_info.loc[idx, "StartingDate"],
            freq=pandas_freq,
            periods=target.shape[0],
        )
        test_dict[idx] = pd.DataFrame({"target": target, "start": start})

    train_data = pandas.PandasDataset(dataframes=train_dict, timestamp="start")
    test_data = pandas.PandasDataset(dataframes=test_dict, timestamp="start")

    meta = MetaData(
        **metadata(
            cardinality=len(train_df),
            freq=pandas_freq,
            prediction_length=prediction_length,
        )
    )

    dataset = TrainDatasets(metadata=meta, train=train_data, test=test_data)
    dataset.save(
        path_str=str(dataset_path), writer=dataset_writer, overwrite=True
    )
