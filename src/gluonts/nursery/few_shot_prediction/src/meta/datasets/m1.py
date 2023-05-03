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
from typing import Optional
import numpy as np
import pandas as pd

from gluonts import json
from gluonts.dataset.repository._m3 import check_dataset
from gluonts.dataset.repository._m3 import M3Setting as M1Setting
from gluonts.dataset.repository._util import metadata, save_to_file, to_dict

from meta.datasets.gluonts import GluonTSDataModule
from meta.datasets.registry import register_data_module


@register_data_module
class M1DataModule(GluonTSDataModule):
    """
    A data module which provides a frequency-category split of the M1 dataset as a standalone dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freq, self.category = self.dataset_name.split("_")[1:]

    def _materialize(self, directory: Path) -> None:
        generate_m1_dataset(
            dataset_path=directory / self.dataset_name,
            m1_freq=self.freq,
            category=self.category,
        )

    @classmethod
    def name(cls) -> str:
        return "dm_m1"


def generate_m1_dataset(
    dataset_path: Path,
    m1_freq: str,
    category: str,
    prediction_length: Optional[int] = None,
):
    from gluonts.dataset.repository import default_dataset_path

    m1_xls_path = default_dataset_path / "MC1001.xls"
    if not os.path.exists(m1_xls_path):
        raise RuntimeError(
            f"The M1 data is available at https://forecasters.org/resources/time-series-data/m-competition/"
            f"Please download the file and copy the files to this location: {m1_xls_path}"
        )

    subsets = {
        "yearly": M1Setting("M1Year", 6, "Y"),
        "quarterly": M1Setting("M1Quart", 8, "Q"),
        "monthly": M1Setting("M1Month", 18, "M"),
    }
    assert (
        m1_freq.lower() in subsets
    ), f"invalid m1_freq='{m1_freq}'. Allowed values: {subsets.keys()}"

    subset = subsets[m1_freq.lower()]
    df = pd.read_excel(m1_xls_path, sheet_name="MC1001")

    def truncate_trailing_nan(v: np.ndarray):
        last_finite_index = np.where(np.isfinite(v))[0][-1]
        return v[: last_finite_index + 1]

    train_data = []
    test_data = []

    def normalize_str(c: str):
        return c.strip().lower()

    # select category
    df["Category"] = df["Category"].apply(normalize_str)
    categories = list(df["Category"].unique())
    assert category in categories, f"category must be one of {categories}"
    df = df.loc[df["Category"] == category]

    # select frequency
    df["Type"] = df["Type"].apply(normalize_str)
    frequencies = list(df["Type"].unique())
    assert m1_freq in frequencies, f"frequency must be one of {frequencies}"
    df = df.loc[df["Type"] == m1_freq]

    i = 0
    for _, row in df.iterrows():
        vals = row.values
        series, n, seasonality, nf, freq, starting_date, category = vals[:7]
        target = np.asarray(vals[7:], dtype=np.float64)
        target = truncate_trailing_nan(target)
        assert len(target) == n + nf
        assert nf == subset.prediction_length

        starting_date = str(starting_date)
        # correct some date errors
        if starting_date.startswith("19.."):
            # use mock date for invalid dates
            starting_date = "1900"
        if starting_date.strip().endswith("JUI"):
            # set to June (?)
            starting_date = starting_date[:-3] + "JUN"
        if starting_date.strip().endswith("JUJUN"):
            starting_date = starting_date[:-5] + "JUN"
        if starting_date.strip().endswith("AVR"):
            starting_date = starting_date[:-3] + "APR"
        if starting_date.strip().endswith("AVJAN"):
            starting_date = starting_date[:-5] + "JAN"
        if starting_date.strip().endswith("DE"):
            starting_date = starting_date[:-3] + "DEC"
        if starting_date.strip().endswith("DEDEC"):
            starting_date = starting_date[:-5] + "DEC"
        if starting_date.strip().endswith("AVAPR"):
            starting_date = starting_date[:-5] + "APR"
        s = pd.Timestamp(starting_date, freq=subset.freq)
        s = s.freq.rollforward(s)
        start = str(s).split(" ")[0]

        d_train = to_dict(
            target_values=target[: -subset.prediction_length],
            start=start,
            item_id=series,
        )
        train_data.append(d_train)

        d_test = to_dict(
            target_values=target,
            start=start,
            item_id=series,
        )
        test_data.append(d_test)
        i += 1

    os.makedirs(dataset_path, exist_ok=True)
    with open(dataset_path / "metadata.json", "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=len(train_data),
                    freq=subset.freq,
                    prediction_length=prediction_length
                    or subset.prediction_length,
                )
            )
        )

    train_file = dataset_path / "train" / "data.json"
    test_file = dataset_path / "test" / "data.json"

    save_to_file(train_file, train_data)
    save_to_file(test_file, test_data)

    check_dataset(dataset_path, len(df), subset.sheet_name)
