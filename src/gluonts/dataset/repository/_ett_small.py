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
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets


# Currently data from only two regions are made public.
NUM_REGIONS = 2


def generate_ett_small_dataset(
    dataset_path: Path,
    dataset_writer: DatasetWriter,
    base_file_name: str,
    freq: str,
    prediction_length: int,
):
    dfs = []
    for i in range(NUM_REGIONS):
        df = pd.read_csv(
            f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset"
            f"/main/ETT-small/{base_file_name}{i+1}.csv"
        )
        df["date"] = df["date"].astype("datetime64[ms]")
        dfs.append(df)

    test = []
    for region, df in enumerate(dfs):
        start = pd.Period(df["date"][0], freq=freq)
        for col in df.columns:
            if col in ["date"]:
                continue
            test.append(
                {
                    "start": start,
                    "item_id": f"{col}_{region}",
                    "target": df[col].values,
                }
            )

    train = []
    for region, df in enumerate(dfs):
        start = pd.Period(df["date"][0], freq=freq)
        for col in df.columns:
            if col in ["date"]:
                continue
            train.append(
                {
                    "start": start,
                    "item_id": f"{col}_{region}",
                    "target": df[col].values[:-prediction_length],
                }
            )

    metadata = MetaData(freq=freq, prediction_length=prediction_length)
    dataset = TrainDatasets(metadata=metadata, train=train, test=test)
    dataset.save(str(dataset_path), writer=dataset_writer, overwrite=True)
