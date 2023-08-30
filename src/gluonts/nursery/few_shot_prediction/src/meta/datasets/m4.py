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
import os
from pathlib import Path
import numpy as np
import pandas as pd

from gluonts.dataset.repository._util import metadata, save_to_file, to_dict

from meta.datasets.gluonts import GluonTSDataModule
from meta.datasets.registry import register_data_module


M4_PREDICTION_LENGTHS = {
    "Hourly": 48,
    "Daily": 14,
    "Weekly": 13,
    "Monthly": 18,
    "Quarterly": 8,
    "Yearly": 6,
}


@register_data_module
class M4DataModule(GluonTSDataModule):
    """
    A data module which provides a frequency-category split of the M4 dataset as a standalone dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freq, self.category = [
            s.capitalize() for s in self.dataset_name.split("_")[1:]
        ]

    def _materialize(self, directory: Path) -> None:
        generate_m4_dataset(
            dataset_path=directory / self.dataset_name,
            m4_freq=self.freq,
            pandas_freq=self.freq[0],
            prediction_length=M4_PREDICTION_LENGTHS[self.freq],
            category=self.category,
        )

    @classmethod
    def name(cls) -> str:
        return "dm_m4"


def generate_m4_dataset(
    dataset_path: Path,
    m4_freq: str,
    pandas_freq: str,
    prediction_length: int,
    category: str,
):
    m4_dataset_url = (
        "https://github.com/M4Competition/M4-methods/raw/master/Dataset"
    )
    meta_df = pd.read_csv(f"{m4_dataset_url}/M4-info.csv", index_col=0)
    meta_df = meta_df.loc[meta_df["SP"] == m4_freq]

    train_df = pd.read_csv(
        f"{m4_dataset_url}/Train/{m4_freq}-train.csv", index_col=0
    )
    train_df = train_df.loc[meta_df["category"] == category]

    test_df = pd.read_csv(
        f"{m4_dataset_url}/Test/{m4_freq}-test.csv", index_col=0
    )
    test_df = test_df.loc[meta_df["category"] == category]

    meta_df = meta_df.loc[meta_df["category"] == category]

    os.makedirs(dataset_path, exist_ok=True)

    with open(dataset_path / "metadata.json", "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=len(train_df),
                    freq=pandas_freq,
                    prediction_length=prediction_length,
                )
            )
        )

    train_file = dataset_path / "train" / "data.json"
    test_file = dataset_path / "test" / "data.json"

    train_target_values = [ts[~np.isnan(ts)] for ts in train_df.values]

    test_target_values = [
        np.hstack([train_ts, test_ts])
        for train_ts, test_ts in zip(train_target_values, test_df.values)
    ]

    if m4_freq == "Yearly":
        # some time series have more than 300 years which can not be
        # represented in pandas, this is probably due to a misclassification
        # of those time series as Yearly. We use only those time series with
        # fewer than upper limit items for this reason.
        start_dates = list(meta_df.StartingDate)
        upper_limit = pd.Timestamp.today().year - min(
            [pd.Timestamp(s).year for s in start_dates]
        )
        filter_long = [len(ts) <= upper_limit for ts in test_target_values]
        train_target_values = [
            ts for ts, b in zip(train_target_values, filter_long) if b
        ]
        test_target_values = [
            ts for ts, b in zip(test_target_values, filter_long) if b
        ]
        meta_df = meta_df.loc[filter_long]

    start_dates = list(meta_df.StartingDate)
    start_dates = [
        sd
        if pd.Timestamp(sd) <= pd.Timestamp("2022")
        else str(pd.Timestamp("1900"))
        for sd in start_dates
    ]

    save_to_file(
        train_file,
        [
            to_dict(
                target_values=target,
                start=start_date,
                cat=[cat],
                item_id=cat,
            )
            for cat, (target, start_date) in enumerate(
                zip(train_target_values, start_dates)
            )
        ],
    )

    save_to_file(
        test_file,
        [
            to_dict(
                target_values=target,
                start=start_date,
                cat=[cat],
                item_id=cat,
            )
            for cat, (target, start_date) in enumerate(
                zip(test_target_values, start_dates)
            )
        ],
    )
