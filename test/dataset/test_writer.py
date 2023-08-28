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

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from gluonts.dataset.common import FileDataset
from gluonts.dataset.arrow import ArrowWriter, ParquetWriter
from gluonts.dataset.jsonl import JsonLinesWriter
from gluonts.dataset.repository import get_dataset


@pytest.mark.skipif(sys.version_info < (3, 7), reason="Requires PyArrow v8.")
@pytest.mark.parametrize(
    "writer", [ArrowWriter(), ParquetWriter(), JsonLinesWriter()]
)
def test_dataset_writer(writer):
    dataset = get_dataset("constant")

    with TemporaryDirectory() as temp_dir:
        writer.write_to_folder(dataset.train, Path(temp_dir))

        loaded = FileDataset(Path(temp_dir), freq="h")
        assert len(dataset.train) == len(loaded)
