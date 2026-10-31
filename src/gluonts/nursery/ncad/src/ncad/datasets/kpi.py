# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from typing import Union
from pathlib import PosixPath

import numpy as np
import pandas as pd

from ncad.ts import TimeSeries, TimeSeriesDataset

from tqdm import tqdm


def kpi(path: Union[PosixPath, str], *args, **kwargs) -> TimeSeriesDataset:
    """

    Args:
        path : Path to the directory containing the two files (.csv and .hdf) with the dataset.

    Source:
        https://github.com/NetManAIOps/KPI-Anomaly-Detection
    """
    print("Loading KPI datasets...")
    path = PosixPath(path).expanduser()
    assert path.is_dir()

    # Verify that all files exist
    files_kpi = ["phase2_train.csv", "phase2_ground_truth.hdf"]
    assert np.all([fn in os.listdir(path) for fn in files_kpi])

    # Load data
    train_df = pd.read_csv(path / files_kpi[0])
    test_df = pd.read_hdf(path / files_kpi[1])

    # transform 'KPI ID' column to string
    train_df["KPI ID"] = train_df["KPI ID"].astype(str)
    test_df["KPI ID"] = test_df["KPI ID"].astype(str)

    kpi_ids = train_df["KPI ID"].unique()
    kpi_ids.sort()
    kpi_ids_test = test_df["KPI ID"].unique()
    kpi_ids_test.sort()

    assert np.all(kpi_ids == kpi_ids_test)

    train_dataset = TimeSeriesDataset()
    test_dataset = TimeSeriesDataset()

    for id_i in tqdm(kpi_ids):

        train_df_i = train_df[train_df["KPI ID"] == id_i].copy()
        train_df_i = train_df_i.sort_values("timestamp").reset_index(drop=True)

        test_df_i = test_df[test_df["KPI ID"] == id_i].copy()
        test_df_i = test_df_i.sort_values("timestamp").reset_index(drop=True)

        train_dataset.append(
            TimeSeries(
                values=train_df_i["value"].to_numpy(),
                labels=train_df_i["label"].to_numpy(),
                item_id=f"{id_i}_train",
            )
        )
        test_dataset.append(
            TimeSeries(
                values=test_df_i["value"].to_numpy(),
                labels=test_df_i["label"].to_numpy(),
                item_id=f"{id_i}_test",
            )
        )

    return train_dataset, test_dataset
