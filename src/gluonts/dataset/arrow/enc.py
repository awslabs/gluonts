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

from dataclasses import dataclass, field
from functools import singledispatch
from itertools import chain
from pathlib import Path
from typing import List, Set, Optional

from toolz.curried import keyfilter, valmap

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gluonts.dataset import Dataset, DatasetWriter
from gluonts.itertools import batcher, rows_to_columns


@dataclass
class ArrowEncoder:
    columns: List[str]
    ndarray_columns: Set[str] = field(default_factory=set)
    flatten_arrays: bool = True

    @classmethod
    def infer(cls, sample: dict, flatten_arrays=True):
        columns = []
        ndarray_columns = set()

        for name, value in sample.items():
            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    ndarray_columns.add(name)

            columns.append(name)

        return cls(
            columns=columns,
            ndarray_columns=ndarray_columns,
            flatten_arrays=flatten_arrays,
        )

    def encode(self, entry: dict):
        result = {}

        for column in self.columns:
            value = entry[column]

            # We need to handle arrays with more than 1 dimension specially.
            # If we don't, pyarrow complains. As an optimisation, we flatten
            # the array to 1d and store its shape to gain zero-copy reads
            # during decoding.
            if column in self.ndarray_columns:
                if self.flatten_arrays:
                    result[f"{column}._np_shape"] = list(value.shape)
                    value = value.flatten()
                else:
                    value = list(value)

            result[column] = value

        return result


def into_arrow_batches(dataset, batch_size=1024, flatten_arrays=True):
    stream = iter(dataset)
    # peek 1
    first = next(stream)
    # re-assemble
    stream = chain([first], stream)

    encoder = ArrowEncoder.infer(first, flatten_arrays=flatten_arrays)
    encoded = map(encoder.encode, stream)

    row_batches = batcher(encoded, batch_size)
    column_batches = map(rows_to_columns, row_batches)

    for batch in column_batches:
        yield pa.record_batch(list(batch.values()), names=list(batch.keys()))


@singledispatch
def _encode_py_to_arrow(val):
    return val


@_encode_py_to_arrow.register
def _encode_py_pd_preiod(val: pd.Period):
    return val.to_timestamp()


def write_dataset(
    Writer, dataset, path, metadata=None, batch_size=1024, flatten_arrays=True
):
    dataset = map(keyfilter(lambda key: key != "source"), dataset)
    dataset = map(valmap(_encode_py_to_arrow), dataset)

    batches = into_arrow_batches(
        dataset, batch_size, flatten_arrays=flatten_arrays
    )
    first = next(batches)

    schema = first.schema

    if metadata is not None:
        schema = schema.with_metadata(metadata)

    with open(path, "wb") as fobj:
        writer = Writer(fobj, schema=schema)
        for batch in chain([first], batches):
            writer.write_batch(batch)

        writer.close()


@dataclass
class ArrowWriter(DatasetWriter):
    stream: bool = False
    suffix: str = ".arrow"
    flatten_arrays: bool = True
    metadata: Optional[dict] = None

    def write_to_file(self, dataset: Dataset, path: Path) -> None:
        if self.stream:
            writer = pa.RecordBatchStreamWriter
        else:
            writer = pa.RecordBatchFileWriter

        write_dataset(
            writer,
            dataset,
            path,
            self.metadata,
            flatten_arrays=self.flatten_arrays,
        )

    def write_to_folder(
        self, dataset: Dataset, folder: Path, name: Optional[str] = None
    ) -> None:
        if name is None:
            name = "data"

        self.write_to_file(dataset, (folder / name).with_suffix(self.suffix))


@dataclass
class ParquetWriter(DatasetWriter):
    suffix: str = ".parquet"
    flatten_arrays: bool = True
    metadata: Optional[dict] = None

    def write_to_file(self, dataset: Dataset, path: Path) -> None:
        write_dataset(
            pq.ParquetWriter,
            dataset,
            path,
            self.metadata,
            flatten_arrays=self.flatten_arrays,
        )

    def write_to_folder(
        self, dataset: Dataset, folder: Path, name: Optional[str] = None
    ) -> None:
        if name is None:
            name = "data"

        self.write_to_file(dataset, (folder / name).with_suffix(self.suffix))
