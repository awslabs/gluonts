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


import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from .dec import ArrowDecoder


class File:
    SUFFIXES = {".parquet", ".arrow"}

    @staticmethod
    def infer(
        path: Path,
    ) -> Union["ArrowFile", "ArrowStreamFile", "ParquetFile"]:
        """Return `ArrowFile`, `ArrowStreamFile` or `ParquetFile` by
        inspecting provided path.

        Arrow's `random-access` format starts with `ARROW1`, so we peek the
        provided file for it.
        """
        with open(path, "rb") as in_file:
            peek = in_file.read(6)

        if peek == b"ARROW1":
            return ArrowFile(path)
        elif peek.startswith(b"PAR1"):
            return ParquetFile(path)
        else:
            return ArrowStreamFile(path)

    @abc.abstractmethod
    def metadata(self) -> Dict[str, str]:
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @abc.abstractmethod
    def __len__(self):
        ...


@dataclass
class ArrowFile(File):
    path: Path
    reader: pa.RecordBatchFileReader = field(init=False)
    decoder: ArrowDecoder = field(init=False)
    _batch_offsets: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )

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
        if len(self.batch_offsets) > 0:
            return self.batch_offsets[-1]

        # empty file
        return 0

    def __iter__(self):
        for batch in self.iter_batches():
            yield from self.decoder.decode_batch(batch)

    def __getitem__(self, idx):
        batch_no, batch_idx = self.location_for(idx)
        return self.decoder.decode(self.reader.get_batch(batch_no), batch_idx)


@dataclass
class ArrowStreamFile(File):
    path: Path
    _decoder: Optional[ArrowDecoder] = field(default=None, init=False)

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
class ParquetFile(File):
    path: Path
    reader: pq.ParquetFile = field(init=False)

    def __post_init__(self):
        self.reader = pq.ParquetFile(self.path)
        self.decoder = ArrowDecoder.from_schema(self.reader.schema_arrow)

    def metadata(self) -> Dict[str, str]:
        metadata = self.reader.schema_arrow.metadata
        if metadata is None:
            return {}

        return {
            key.decode(): value.decode() for key, value in metadata.items()
        }

    def __iter__(self):
        for batch in self.reader.iter_batches():
            yield from self.decoder.decode_batch(batch)

    def __len__(self):
        # One would think that pq.ParquetFile had a nicer way to get its length
        return self.reader.metadata.num_rows
