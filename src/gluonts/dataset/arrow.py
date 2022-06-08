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

Fast and efficient datasets using `pyarrow`.

This module provides three datasets:

    * ``ArrowDataset`` (arrow random-access binary format)
    * ``ArrowStreamDataset`` (arrow streaming binary format)
    * ``ParquetDataset``

"""

from dataclasses import dataclass, field
from functools import singledispatch
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from gluonts.itertools import batcher, rows_to_columns


@singledispatch
def _arrow_to_py(scalar):
    """Convert arrow scalar value to python value."""

    raise NotImplementedError(scalar, scalar.__class__)


@_arrow_to_py.register
def _arrow_to_py_scalar(scalar: pa.Scalar):
    return scalar.as_py()


@_arrow_to_py.register
def _arrow_to_py_list_scalar(scalar: pa.ListScalar):
    arr = scalar.values.to_numpy(zero_copy_only=False)

    if arr.dtype == object:
        arr = np.array(list(arr))

    return arr


def into_arrow_batches(dataset, batch_size=1024, flatten_arrays=True):
    stream = iter(dataset)
    # peak 1
    first = next(stream)
    # re-assemble
    stream = chain([first], stream)

    encoder = ArrowEncoder.infer(first, flatten_arrays=flatten_arrays)
    encoded = map(encoder.encode, stream)

    row_batches = batcher(encoded, batch_size)
    column_batches = map(rows_to_columns, row_batches)

    for batch in column_batches:
        yield pa.record_batch(list(batch.values()), names=list(batch.keys()))


@dataclass
class ArrowDecoder:
    columns: Dict[str, int]
    ndarray_columns: Dict[str, int]

    @classmethod
    def from_schema(cls, schema):
        columns = {}
        ndarray_columns = {}

        for idx, column in enumerate(schema):
            if column.name.endswith("._np_shape"):
                ndarray_columns[(column.name.rsplit(".", 1)[0])] = idx
            else:
                columns[column.name] = idx

        return cls(columns, ndarray_columns)

    def decode(self, batch, row_number):
        for row in self.decode_batch(batch.slice(row_number, row_number + 1)):
            return row

    def decode_batch(self, batch):
        rows = zip(*batch)

        for raw_row in rows:
            row = {}
            for column_name, column_idx in self.columns.items():
                value = _arrow_to_py(raw_row[column_idx])

                shape_idx = self.ndarray_columns.get(column_name)

                if shape_idx is not None:
                    shape = _arrow_to_py(raw_row[shape_idx])
                    value = value.reshape(shape)

                row[column_name] = value

            yield row


@dataclass
class ArrowDataset:
    path: Path
    reader: pa.RecordBatchFileReader = field(init=False)
    decoder: ArrowDecoder = field(init=False)
    _batch_offsets: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )

    @staticmethod
    def create(
        dataset, path, metadata=None, chunk_size=1024, flatten_arrays=True
    ):
        batches = into_arrow_batches(
            dataset, chunk_size, flatten_arrays=flatten_arrays
        )
        first = next(batches)

        schema = first.schema

        if metadata is not None:
            schema = schema.with_metadata(metadata)

        with open(path, "wb") as fobj:
            writer = pa.RecordBatchFileWriter(fobj, schema=schema)
            for batch in chain([first], batches):
                writer.write_batch(batch)

            writer.close()

    @property
    def metadata(self) -> Dict[str, str]:
        metadata = self.reader.schema.metadata
        if metadata is None:
            return {}

        return {
            key.decode(): value.decode() for key, value in metadata.items()
        }

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
    _decoder: Optional[ArrowDecoder] = field(default=None, init=False)

    @staticmethod
    def create(
        dataset, path, metadata=None, chunk_size=1024, flatten_arrays=True
    ):
        batches = into_arrow_batches(
            dataset, chunk_size, flatten_arrays=flatten_arrays
        )
        first = next(batches)

        schema = first.schema

        if metadata is not None:
            schema = schema.with_metadata(metadata)

        with open(path, "wb") as fobj:
            writer = pa.RecordBatchStreamWriter(fobj, schema=schema)
            for batch in chain([first], batches):
                writer.write_batch(batch)

            writer.close()

    @property
    def metadata(self) -> Dict[str, str]:
        with open(self.path, "rb") as infile:
            metadata = pa.RecordBatchStreamReader(infile).schema.metadata

        if metadata is None:
            return {}

        return {
            key.decode(): value.decode() for key, value in metadata.items()
        }

    def __iter__(self):
        with open(self.path, "rb") as infile:
            reader = pa.RecordBatchStreamReader(infile)
            if self._decoder is None:
                self._decoder = ArrowDecoder.from_schema(reader.schema)

            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    return

                yield from self._decoder.decode_batch(batch)

    def __len__(self):
        return sum(1 for _ in self)


@dataclass
class ParquetDataset:
    path: Path
    reader: pa.RecordBatchFileReader = field(init=False)
    _length: Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        self.reader = pq.ParquetFile(self.path)
        self.decoder = ArrowDecoder.from_schema(self.reader.schema)

    @property
    def metadata(self) -> Dict[str, str]:
        metadata = self.reader.schema.metadata
        if metadata is None:
            return {}

        return {
            key.decode(): value.decode() for key, value in metadata.items()
        }

    def __iter__(self):
        for batch in self.reader.iter_batches():
            yield from self.decoder.decode_batch(batch)

    def __len__(self):
        if self._length is None:
            self._length = self.reader.scan_contents()

        return self._length


# TODO: We could use pyarrow to load json-lines datasets. However, it appears
# slower than our custom implementation and this doesn't support streaming.
# So maybe it's better to just remove this.
# @dataclass
# class JsonDataset:
#     path: Path
#     table: pa.Table = field(init=False)
#     _decoder: Optional[ArrowDecoder] = field(default=None, init=False)

#     def __post_init__(self):
#         self.table = pyarrow.json.read_json(
#             self.path, pyarrow.json.ReadOptions(block_size=1024**3)
#         )
#         self._decoder = ArrowDecoder.from_schema(self.table.schema)

#     def __iter__(self):
#         yield from self._decoder.decode_batch(self.table)

#     def __getitem__(self, idx):
#         return self._decoder.decode(self.table, idx)

#     def __len__(self):
#         return self.table.num_rows


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


def infer_arrow_dataset(path: Path) -> Union[ArrowDataset, ArrowStreamDataset]:
    """Return either `ArrowDataset` or `ArrowStreamDataset` by inspecting
    provided path.

    Arrow's `random-access` format starts with `ARROW1`, so we peak the
    provided file for it.
    """
    with open(path, "rb") as in_file:
        peak = in_file.read(6)

    if peak == b"ARROW1":
        return ArrowDataset(path)
    else:
        return ArrowStreamDataset(path)
