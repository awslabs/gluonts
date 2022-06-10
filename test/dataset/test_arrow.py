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

import numpy as np
from numpy.testing import assert_equal
import pytest

from gluonts.dataset.arrow import (
    ArrowFile,
    ArrowStreamFile,
    ParquetFile,
    write_dataset,
)


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


@pytest.mark.parametrize("File", [ArrowFile, ArrowStreamFile, ParquetFile])
@pytest.mark.parametrize("flatten_arrays", [True, False])
def test_arrow(File, flatten_arrays):
    data = make_data(5)

    with tempfile.TemporaryDirectory() as path:
        path = Path(path, "data.arrow")

        # create file on disk
        write_dataset(
            File,
            data,
            path,
            metadata={"freq": "H"},
            flatten_arrays=flatten_arrays,
        )

        dataset = File.infer(path)

        assert len(data) == len(dataset)
        assert dataset.metadata()["freq"] == "H"

        for orig, arrow_value in zip(data, dataset):
            print("O", orig)
            print("P", arrow_value)
            assert_equal(orig, arrow_value)
