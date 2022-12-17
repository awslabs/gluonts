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

from datetime import datetime
import functools
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Optional, BinaryIO

import numpy as np
import pandas as pd
from toolz import valmap

from gluonts import json
from gluonts.exceptions import GluonTSDataError

from . import Dataset, DatasetWriter


def load(file_obj):
    return map(json.loads, file_obj)


def dump(objects, file_obj):
    for object_ in objects:
        json.dump(object_, file_obj, nl=True)


@functools.singledispatch
def encode_json(arg):
    if isinstance(arg, (str, int)):
        return arg

    if isinstance(arg, float):
        if np.isnan(arg):
            return "NaN"
        elif np.isposinf(arg):
            return "Infinity"
        elif np.isneginf(arg):
            return "-Infinity"
        return arg

    if isinstance(arg, datetime):
        return str(arg)

    raise ValueError(f"Can't encode {arg!r}")


@encode_json.register(dict)
def _encode_json_dict(arg: dict):
    return valmap(encode_json, arg)


@encode_json.register(list)
def _encode_json_list(arg: list):
    return list(map(encode_json, arg))


@encode_json.register(np.ndarray)
def _encode_json_array(arg: np.ndarray):
    if np.issubdtype(arg.dtype, int):
        return arg.tolist()

    if np.issubdtype(arg.dtype, np.floating):
        b = np.array(arg, dtype=object)
        b[np.isnan(arg)] = "Nan"
        b[np.isposinf(arg)] = "Infinity"
        b[np.isneginf(arg)] = "-Infinity"

        return b.tolist()

    return _encode_json_list(arg.tolist())


@encode_json.register(pd.Period)
def _encode_json_period(arg: pd.Period):
    return str(arg)


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
                        f"Could not read json line {line_number}, {line}"
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


@dataclass
class JsonLinesWriter(DatasetWriter):
    use_gzip: bool = True
    suffix: str = ".json"
    # Python uses `compresslevel=9` by default, which is very slow
    # We opt for faster writes by default, for more modest size savings
    compresslevel: int = 4

    def write_to_file(self, dataset: Dataset, path: Path) -> None:
        if self.use_gzip:
            out_file = cast(
                BinaryIO,
                gzip.open(path, "wb", compresslevel=self.compresslevel),
            )
        else:
            out_file = open(path, "wb")

        with out_file:
            for entry in dataset:
                json.bdump(encode_json(entry), out_file, nl=True)

    def write_to_folder(
        self, dataset: Dataset, folder: Path, name: Optional[str] = None
    ) -> None:
        if name is None:
            name = "data"

        if self.use_gzip:
            suffix = self.suffix + ".gz"
        else:
            suffix = self.suffix

        self.write_to_file(dataset, (folder / name).with_suffix(suffix))
