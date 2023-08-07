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


from gluonts.dataset.common import FileDataset
from gluonts.dataset.jsonl import JsonLinesWriter, JsonLinesFile

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


def test_jsonl_slice():
    data = list(range(10))

    with tempfile.TemporaryDirectory() as path:
        tmp_file = Path(path) / "data.json"

        JsonLinesWriter(use_gzip=False).write_to_file(data, tmp_file)

        reader = JsonLinesFile(tmp_file)

        assert len(reader) == len(data)
        assert list(reader) == data

        assert reader[0] == data[0]
        assert reader[-1] == data[-1]
        assert sum(reader) == sum(data)

        assert list(reader[:5]) == data[:5]
        assert len(reader[:5]) == len(data[:5])

        assert list(reader[10:]) == data[10:]
        assert len(reader[10:]) == len(data[10:])

        assert list(reader[3:7]) == data[3:7]
        assert len(reader[3:7]) == len(data[3:7])
