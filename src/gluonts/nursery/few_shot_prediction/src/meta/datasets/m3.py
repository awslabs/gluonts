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

from typing import Optional
import warnings
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from gluonts.dataset.repository._m3 import M3Setting, check_dataset
from gluonts.dataset.repository._util import metadata, save_to_file, to_dict

from meta.datasets.gluonts import GluonTSDataModule
from meta.datasets.registry import register_data_module


@register_data_module
class M3DataModule(GluonTSDataModule):
    """
    A data module which provides a frequency-category split of the M3 dataset as a standalone dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freq, self.category = self.dataset_name.split("_")[1:]

    def _materialize(self, directory: Path) -> None:
        generate_m3_dataset(
            dataset_path=directory / self.dataset_name,
            m3_freq=self.freq,
            category=self.category,
        )

    @classmethod
    def name(cls) -> str:
        return "dm_m3"


def generate_m3_dataset(
    dataset_path: Path,
    m3_freq: str,
    category: str,
    prediction_length: Optional[int] = None,
):
    from gluonts.dataset.repository import default_dataset_path

    m3_xls_path = default_dataset_path / "M3C.xls"
    if not os.path.exists(m3_xls_path):
        raise RuntimeError(
            f"The m3 data is available at https://forecasters.org/resources/time-series-data/m3-competition/ "
            f"Please download the file and copy the files to this location: {m3_xls_path}"
        )

    subsets = {
        "yearly": M3Setting("M3Year", 6, "Y"),
        "quarterly": M3Setting("M3Quart", 8, "Q"),
        "monthly": M3Setting("M3Month", 18, "M"),
        "other": M3Setting("M3Other", 8, "Q"),
    }
    assert (
        m3_freq.lower() in subsets
    ), f"invalid m3_freq='{m3_freq}'. Allowed values: {subsets.keys()}"

    if m3_freq.lower() == "other":
        warnings.warn(
            "Be aware: The M3-other dataset does not have a known frequency. Since gluonts needs a known frequency, "
            "we will generate the dataset with an artificial `quarterly` frequency."
        )

    subset = subsets[m3_freq.lower()]
    df = pd.read_excel(m3_xls_path, sheet_name=subset.sheet_name)

    def truncate_trailing_nan(v: np.ndarray):
        last_finite_index = np.where(np.isfinite(v))[0][-1]
        return v[: last_finite_index + 1]

    train_data = []
    test_data = []

    def normalize_category(c: str):
        return c.strip().lower()

    df["Category"] = df["Category"].apply(normalize_category)
    categories = list(df["Category"].unique())
    assert category in categories, f"category must be one of {categories}"
    df = df.loc[df["Category"] == category]

    cat_map = {c: i for i, c in enumerate(categories)}

    i = 0
    for _, row in df.iterrows():
        vals = row.values
        series, n, nf, category, starting_year, starting_offset = vals[:6]
        target = np.asarray(vals[6:], dtype=np.float64)
        target = truncate_trailing_nan(target)
        assert len(target) == n
        assert nf == subset.prediction_length
        mock_start = "1750"

        if starting_year == 0:
            assert starting_offset == 0
            starting_year = mock_start

        # fix bugs in M3C xls
        if series == "N1071":
            # bug in the m3 dataset
            starting_offset = 1
        if series == "N 184":
            starting_offset = 1

        offset = max(starting_offset - 1, 0)

        if subset.freq == "Q":
            assert 0 <= offset < 4
            time_stamp = f"{starting_year}-{3 * (offset + 1):02}-15"
        elif subset.freq == "Y":
            assert offset == 0
            time_stamp = f"{starting_year}-12-15"
        elif subset.freq == "M":
            assert 0 <= offset < 12
            time_stamp = f"{starting_year}-{offset + 1:02}-15"

        s = pd.Timestamp(time_stamp, freq=subset.freq)
        s = s.freq.rollforward(s)
        start = str(s).split(" ")[0]
        cat = [i, cat_map[category]]

        d_train = to_dict(
            target_values=target[: -subset.prediction_length],
            start=start,
            cat=cat,
            item_id=series,
        )
        train_data.append(d_train)

        d_test = to_dict(
            target_values=target, start=start, cat=cat, item_id=series
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
