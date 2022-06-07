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
import pyarrow.parquet as pq

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
    flatten_arrays: bool = True

    @classmethod
    def infer(cls, sample: dict, flatten_arrays=True):
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
            flatten_arrays=flatten_arrays,
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
                if self.flatten_arrays:
                    result[shape_column_of(column)] = list(value.shape)
                    value = value.flatten()
                else:
                    value = list(value)

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
    arr = scalar.values.to_numpy(zero_copy_only=False)

    if arr.dtype == object:
        arr = np.array(list(arr))

    return arr


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

        columns = [chunk[0] for chunk in columns]

        for column_name, column_idx in self.columns.items():
            value = scalar_to_py(columns[column_idx])

            shape_idx = self.ndarray_columns.get(column_name)
            if shape_idx is not None:
                shape = scalar_to_py(columns[shape_idx])
                value.reshape(shape)

            row[column_name] = value

        return row


@dataclass
class ArrowRandomAccessDataset:
    path: Path
    reader: pa.RecordBatchFileReader = field(init=False)
    decoder: ArrowDecoder = field(init=False)
    _batch_offsets: np.array = None

    @property
    def batch_offsets(self):
        if self._batch_offsets is None:
            self._batch_offsets = np.cumsum(
                list(map(len, self.iter_batches()))
            )

        return self._batch_offsets

    def __post_init__(self):
        self.reader = pa.RecordBatchFileReader(self.path)
        self.decoder = ArrowDecoder.from_schema(self.schema)

    def location_for(self, idx):
        batch_no = np.searchsorted(self.batch_offsets, idx)
        if batch_no == 0:
            batch_idx = idx
        else:
            batch_idx = idx - self.batch_offsets[batch_no - 1]
        return batch_no, batch_idx

    @property
    def schema(self):
        return self.reader.schema

    def iter_batches(self):
        for batch_no in range(self.reader.num_record_batches):
            yield self.reader.get_batch(batch_no)

    def __len__(self):
        return sum(self.batch_offsets)

    def __iter__(self):
        for batch in self.iter_batches():
            yield from self.decoder.decode_batch(batch)

    def __getitem__(self, idx):
        batch_no, batch_idx = self.location_for(idx)
        return self.decoder.decode(self.reader.get_batch(batch_no), idx)


@dataclass
class ArrowStreamDataset:
    path: Path
    decoder: Optional[ArrowDecoder] = None

    def __iter__(self):
        with open(self.path, "rb") as infile:
            reader = pa.RecordBatchStreamReader(infile)
            if self.decoder is None:
                self.decoder = ArrowDecoder.from_schema(reader.schema)

            while True:
                try:
                    batch = reader.read_next_batch()
                    yield from self.decoder.decode_batch(batch)
                except StopIteration:
                    return

    def __len__(self):
        return sum(1 for _ in self)


@dataclass
class ParquetDataset:
    path: Path

    def __iter__(self):
        pf = pq.ParquetFile(self.path)
        decoder = ArrowDecoder.from_schema(pf.schema)

        for batch in pf.iter_batches():
            yield from decoder.decode_batch(batch)


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


def convert(dataset, chunk_size=1024, flatten_arrays=True):
    dataset = iter(dataset)
    first = next(dataset)

    batcher = ArrowBatcher(
        encoder=ArrowEncoder.infer(
            first,
            flatten_arrays=flatten_arrays,
        ),
        chunk_size=chunk_size,
    )

    yield from batcher.batchify(chain([first], dataset))


def write(dataset, path, chunk_size=1024, flatten_arrays=True):
    batches = convert(dataset, chunk_size, flatten_arrays=flatten_arrays)
    first = next(batches)

    with open(path, "wb") as fobj:
        writer = pa.RecordBatchFileWriter(fobj, schema=first.schema)
        for batch in chain([first], batches):
            writer.write_batch(batch)

        writer.close()


def write_streaming(dataset, path, chunk_size=1024, flatten_arrays=True):
    batches = convert(dataset, chunk_size, flatten_arrays=flatten_arrays)
    first = next(batches)

    with open(path, "wb") as fobj:
        writer = pa.RecordBatchStreamWriter(fobj, schema=first.schema)
        for batch in chain([first], batches):
            writer.write_batch(batch)

        writer.close()


# def write_parquet(dataset, path):


# def write_parquet(dataset, path, chunk_size=1024):
#     batches = convert(dataset, chunk_size)
#     first = next(batches)

#     with open(path, "wb") as fobj:
#         writer = pa.RecordBatchStreamWriter(fobj, schema=first.schema)
#         for batch in chain([first], batches):
#             writer.write_batch(batch)

#         writer.close()
