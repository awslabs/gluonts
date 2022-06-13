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

import functools
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from gluonts import json
from gluonts.exceptions import GluonTSDataError


def load(file_obj):
    return map(json.loads, file_obj)


def dump(objects, file_obj):
    for object_ in objects:
        json.dump(object_, file_obj, nl=True)


@dataclass
class JsonLinesFile:
    """
    An iterable type that draws from a JSON Lines file.

    Parameters
    ----------
    path
        Path of the file to load data from. This should be a valid
        JSON Lines file.
    """

    SUFFIXES = {
        ".json",
        ".json.gz",
        ".jsonl",
        ".jsonl.gz",
    }

    path: Path

    def open(self):
        if self.path.suffix == ".gz":
            return gzip.open(self.path)

        return open(self.path, "rb")

    def __iter__(self):
        with self.open() as jsonl_file:
            for line_number, line in enumerate(jsonl_file):
                try:
                    yield json.loads(line)
                except ValueError:
                    raise GluonTSDataError(
                        f"Could not read json line {line_number}, {raw}"
                    )

    def __len__(self):
        # 1MB
        BUF_SIZE = 1024**2

        with self.open() as file_obj:
            read_chunk = functools.partial(file_obj.read, BUF_SIZE)
            file_len = sum(
                chunk.count(b"\n") for chunk in iter(read_chunk, b"")
            )
            return file_len
