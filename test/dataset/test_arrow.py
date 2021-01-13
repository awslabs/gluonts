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
from pathlib import Path
import platform

import numpy as np
import pytest

from gluonts.dataset.common import ArrowDataset, ArrowWriter, ListDataset


def rand_start():
    year = np.random.randint(low=1900, high=2020)
    month = np.random.randint(low=1, high=13)
    day = np.random.randint(low=1, high=29)
    return f"{year}-{month:02d}-{day:02d}"


def make_data(n: int):
    data = []
    for i in range(n):
        ts_len = np.random.choice([1, 100, 700, 901])
        ts = {
            "start": rand_start(),
            "target": np.random.uniform(size=ts_len).astype(np.float32),
            "feat_dynamic_real": np.random.uniform(size=(3, ts_len)).astype(
                np.float32
            ),
            "feat_static_cat": np.random.randint(
                low=0, high=10, size=7
            ).astype(np.int64),
        }
        data.append(ts)
    return data


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Skip tests on windows, since removing dirs with memory mapped files fails.",
)
def test_arrow():
    n = 30
    data = make_data(n)

    with tempfile.TemporaryDirectory() as path:
        data_arrow_file = Path(path, "data.arrow")
        data_convert_arrow_file = Path(path, "data_convert.arrow")
        with ArrowWriter(data_arrow_file) as aw:
            for d in data:
                aw.write_record(d)
        with ArrowWriter(data_convert_arrow_file) as aw:
            for d in data:
                r = {
                    "start": d["start"],
                    "target": d["target"].astype(float).tolist(),
                    "feat_dynamic_real": d["feat_dynamic_real"]
                    .astype(float)
                    .tolist(),
                    "feat_static_cat": d["feat_static_cat"]
                    .astype(int)
                    .tolist(),
                }
                aw.write_record(r)

        freq = "1min"
        list_ds = ListDataset(data, freq=freq)
        data_list = [d for d in list_ds]
        data_arrow = [
            d for d in ArrowDataset.load_files(data_arrow_file, freq=freq)
        ]
        data_convert_arrow = [
            d
            for d in ArrowDataset.load_files(
                data_convert_arrow_file, freq=freq
            )
        ]

        assert n == len(data_arrow)
        assert n == len(data_convert_arrow)
        for i in range(len(data_list)):
            d = data_list[i]
            d1 = data_arrow[i]
            d2 = data_convert_arrow[i]
            assert d["start"] == d1["start"]
            assert d["start"] == d2["start"]
            assert np.all(d["target"] == d1["target"])
            assert np.all(d["target"] == d2["target"])
            assert np.all(d["feat_dynamic_real"] == d1["feat_dynamic_real"])
            assert np.all(d["feat_dynamic_real"] == d2["feat_dynamic_real"])
            assert np.all(d["feat_static_cat"] == d1["feat_static_cat"])
            assert np.all(d["feat_static_cat"] == d2["feat_static_cat"])
