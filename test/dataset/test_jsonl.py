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

import gzip
import tempfile
from pathlib import Path

import pytest

from gluonts.dataset.common import FileDataset

N = 3

data = [
    '{"start": "2014-09-07", "target": [1, 2, 3]}',
] * N


def test_jsonl():
    with tempfile.TemporaryDirectory() as path:
        with Path(path, "data.json").open("w") as out_file:
            for line in data:
                out_file.write(line + "\n")

        assert len(FileDataset(path, freq="D")) == N
        assert len(list(FileDataset(path, freq="D"))) == N


def test_jsonlgz():
    with tempfile.TemporaryDirectory() as path:
        with gzip.open(Path(path, "data.json.gz"), "wt") as out_file:
            for line in data:
                out_file.write(line + "\n")

        assert len(FileDataset(path, freq="D")) == N
        assert len(list(FileDataset(path, freq="D"))) == N
