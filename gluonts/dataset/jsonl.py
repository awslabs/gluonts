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

# Standard library imports
import functools
from pathlib import Path
from typing import NamedTuple

# Third-party imports
import ujson as json

# First-party imports
from gluonts.core.exception import GluonTSDataError


def load(file_obj):
    for line in file_obj:
        yield json.loads(line)


def dump(objects, file_obj):
    for object_ in objects:
        file_obj.writeline(json.dumps(object_))


class Span(NamedTuple):
    path: Path
    line: int


class Line(NamedTuple):
    content: object
    span: Span


class JsonLinesFile:
    """
    An iterable type that draws from a JSON Lines file.

    Paramters
    ---------
    path
        Path of the file to load data from. This should be a valid
        JSON Lines file.
    """

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path) as jsonl_file:
            for line_number, raw in enumerate(jsonl_file, start=1):
                span = Span(path=self.path, line=line_number)
                try:
                    yield Line(json.loads(raw), span=span)
                except ValueError as _:
                    raise GluonTSDataError(
                        f"Could not read json line {line_number}, {raw}"
                    )

    def __len__(self):
        # 1MB
        BUF_SIZE = 1024 ** 2

        with open(self.path) as file_obj:
            read_chunk = functools.partial(file_obj.read, BUF_SIZE)
            return sum(chunk.count('\n') for chunk in iter(read_chunk, ''))
