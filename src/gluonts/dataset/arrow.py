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


"""
Arrow Dataset
~~~~~~~~~~~~~

This module provides a fast and efficient dataset using `pyarrow`.

Arrow allows us to use zero-copy.

There are two properties of Arrow, which introduce some friction:

1) Arrow is a columnar format, meaning that instead of writing out row after
row(like in a spreadsheet) it writes out column after column -- basically
flipping the table. There are some benefits in using a columnar format, since
values of the same type are stored together which benefits analytical queries.
However, in GluonTS a columnar format doesn't make much sense, since we are
always only interested in handling rows.

2) Interop between NumPy and Arrow is not seemless when using multi-dimensional
arrays. Whilst 1D-arrays can be converted using zero-copy, it doesn't work for
higher dimensional data. Thus, we always store arrays as one-dimensional and
store the shape for higher dimensions to invoke `.reshape(...)` later on to
maintain zero-copy.
"""

from dataclasses import dataclass, field
from functools import singledispatch
from itertools import chain
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
)

import numpy as np
import pandas as pd
import pyarrow as pa

from gluonts.itertools import batcher

# from .common import Dataset, DataEntry


def shape_column_of(column):
    return f"{column}._np_shape"


T = TypeVar("T")


def transpose(data: List[Dict[str, T]]) -> Dict[str, List[T]]:
    if not data:
        return {}

    column_names = list(data[0])

    return {
        column: [entry[column] for entry in data] for column in column_names
    }


@dataclass
class ArrowEncoder:
    columns: List[str]
    convert: Dict[str, Callable] = field(default_factory=set)
    ndarray_columns: Set[str] = field(default_factory=set)

    @classmethod
    def infer(cls, sample: dict):
        columns = []
        ndarray_columns = set()
        convert = {}

        for name, value in sample.items():
            if name == "source":
                continue

            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    ndarray_columns.add(name)

            if isinstance(value, pd.Period):
                convert[name] = pd.Period.to_timestamp

            columns.append(name)

        return cls(
            columns=columns,
            ndarray_columns=ndarray_columns,
            convert=convert,
        )

    def column_names(self):
        return self.columns + list(map(shape_column_of, self.ndarray_columns))

    def encode(self, entry: dict):
        result = {}

        for column in self.columns:
            value = entry[column]

            fn = self.convert.get(column)
            if fn is not None:
                value = fn(value)

            if column in self.ndarray_columns:
                result[shape_column_of(column)] = list(value.shape)
                value = value.flatten()

            result[column] = value

        return result


@singledispatch
def scalar_to_py(scalar):
    raise NotImplementedError(scalar, scalar.__class__)
    # return scalar.as_py()


@scalar_to_py.register
def _(scalar: pa.Scalar):
    return scalar.as_py()


@scalar_to_py.register
def _(scalar: pa.ListScalar):
    return scalar.values.to_numpy()


@dataclass
class ArrowDecoder:
    columns: Dict[str, int]
    ndarray_columns: Dict[str, int]

    @classmethod
    def from_schema(cls, schema):
        columns = {}
        ndarray_columns = {}

        for idx, field in enumerate(schema):
            if field.name.endswith("._np_shape"):
                ndarray_columns[(field.name.rsplit(".", 1)[0])] = idx
            else:
                columns[field.name] = idx

        return cls(columns, ndarray_columns)

    def decode_batch(self, batch):
        for row_number in range(batch.num_rows):
            yield self.decode(batch, row_number)

    def decode(self, batch, row_number):
        row = {}
        columns = batch.slice(row_number, row_number + 1)

        # if isinstance(columns, pa.Table):
        columns = [chunk[0] for chunk in columns]

        for column_name, column_idx in self.columns.items():
            value = scalar_to_py(columns[column_idx])

            shape_idx = self.ndarray_columns.get(column_name)
            if shape_idx is not None:
                shape = scalar_to_py(batch[shape_idx])
                value.reshape(shape)

            row[column_name] = value

        return row


@dataclass
class ArrowTable:
    table: pa.Table
    decoder: ArrowEncoder

    @classmethod
    def from_file(cls, path):
        with open(path, "rb") as infile:
            table = pa.RecordBatchFileReader(infile).read_all()

        return cls.new(table)

    @classmethod
    def new(cls, table: pa.Table):
        return cls(table, ArrowDecoder.from_schema(table.schema))

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        return self.decoder.decode(self.table, idx)


@dataclass
class StreamDataset:
    path: Path

    def __iter__(self):
        with open(self.path, "rb") as infile:
            reader = pa.RecordBatchFileReader(infile)

            for batch_number in range(reader.num_record_batches):
                batch = reader.get_batch(batch_number)
                table = ArrowTable.new(batch)

                yield from table


@dataclass
class ArrowBatcher:
    encoder: ArrowEncoder
    chunk_size: Optional[int] = 1024

    def batchify(self, stream):
        stream = map(self.encoder.encode, stream)
        batches = batcher(stream, self.chunk_size)

        for batch in map(transpose, batches):
            names = list(batch.keys())
            values = list(batch.values())
            yield pa.record_batch(values, names=names)


def convert(dataset, chunk_size=1024):
    dataset = iter(dataset)
    first = next(dataset)

    batcher = ArrowBatcher(
        encoder=ArrowEncoder.infer(first), chunk_size=chunk_size
    )

    yield from batcher.batchify(chain([first], dataset))


def write(dataset, path, chunk_size=1024):
    batches = convert(dataset)
    first = next(batches)

    with open(path, "wb") as fobj:
        writer = pa.RecordBatchFileWriter(fobj, schema=first.schema)
        for batch in chain([first], batches):
            writer.write_batch(batch)

        writer.close()
