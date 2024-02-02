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
from typing import Dict, Optional, Union, List

import numpy as np
from toolz import take, drop, first

import pyarrow as pa
import pyarrow.parquet as pq

from .dec import ArrowDecoder


class File:
    SUFFIXES = {".parquet", ".arrow", ".feather"}

    @staticmethod
    def infer(
        path: Path,
    ) -> Union["ArrowFile", "ArrowStreamFile", "ParquetFile"]:
        """
        Return `ArrowFile`, `ArrowStreamFile` or `ParquetFile` by inspecting
        provided path.

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
    def metadata(self) -> Dict[str, str]: ...

    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __len__(self): ...


@dataclass
class ArrowFile(File):
    path: Path
    reader: pa.RecordBatchFileReader = field(init=False)
    decoder: ArrowDecoder = field(init=False)
    _batch_offsets: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )
    _start: int = 0
    _take: Optional[int] = None

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
        if idx == 0:
            return 0, 0

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
        if self._take is not None:
            return self._take

        if len(self.batch_offsets) > 0:
            return self.batch_offsets[-1] - self._start

        # empty file
        return 0

    def __iter__(self):
        def iter_values():
            # yield from starting batch
            batch_no, batch_idx = self.location_for(self._start)
            sub_batch = self.reader.get_batch(batch_no)[batch_idx:]
            yield from self.decoder.decode_batch(sub_batch)

            for batch_no_ in range(
                batch_no + 1, self.reader.num_record_batches
            ):
                yield from self.decoder.decode_batch(
                    self.reader.get_batch(batch_no_)
                )

        yield from take(self._take, iter_values())

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            assert idx.step is None or idx.step == 1

            # normalize index
            start, stop, _step = idx.indices(len(self))
            idx = slice(start + self._start, stop + self._start)

            return ArrowFile(
                self.path,
                _start=idx.start,
                _take=max(0, idx.stop - idx.start),
            )

        if self._start is not None:
            idx += self._start

        batch_no, batch_idx = self.location_for(idx)
        return self.decoder.decode(self.reader.get_batch(batch_no), batch_idx)


@dataclass
class ArrowStreamFile(File):
    path: Path
    _decoder: Optional[ArrowDecoder] = field(default=None, init=False)
    _start: int = 0
    _take: Optional[int] = None

    def metadata(self) -> Dict[str, str]:
        with open(self.path, "rb") as infile:
            metadata = pa.RecordBatchStreamReader(infile).schema.metadata

        if metadata is None:
            return {}

        return {
            key.decode(): value.decode() for key, value in metadata.items()
        }

    def __iter__(self):
        def iter_values():
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

        yield from take(self._take, drop(self._start, iter_values()))

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            assert idx.step is None or idx.step == 1

            # normalize index
            start, stop, _step = idx.indices(len(self))
            idx = slice(start + self._start, stop + self._start)

            return ArrowStreamFile(
                self.path,
                _start=idx.start,
                _take=max(0, idx.stop - idx.start),
            )

        return first(self[idx:])


@dataclass
class ParquetFile(File):
    path: Path
    reader: pq.ParquetFile = field(init=False)
    _start: int = 0
    _take: Optional[int] = None

    # Note: accumulated
    _row_group_sizes: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.reader = pq.ParquetFile(self.path)
        self.decoder = ArrowDecoder.from_schema(self.reader.schema_arrow)

        if not self._row_group_sizes:
            self._row_group_sizes = np.cumsum(
                [
                    self.reader.metadata.row_group(row_group).num_rows
                    for row_group in range(self.reader.metadata.num_row_groups)
                ]
            )

    def location_for(self, idx):
        if idx == 0:
            return 0, 0

        row_group = np.searchsorted(self._row_group_sizes, idx)
        if row_group == 0:
            row_index = idx
        else:
            row_index = idx - self._row_group_sizes[row_group - 1]
        return row_group, row_index

    def metadata(self) -> Dict[str, str]:
        metadata = self.reader.schema_arrow.metadata
        if metadata is None:
            return {}

        return {
            key.decode(): value.decode() for key, value in metadata.items()
        }

    def __iter__(self):
        def iter_values():
            row_group, row_index = self.location_for(self._start)

            table = self.reader.read_row_group(row_group)
            yield from self.decoder.decode_batch(table[row_index:])

            for row_group_ in range(row_group + 1, len(self._row_group_sizes)):
                table = self.reader.read_row_group(row_group_)
                yield from self.decoder.decode_batch(table)

        yield from take(self._take, iter_values())

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            assert idx.step is None or idx.step == 1

            # normalize index
            start, stop, _step = idx.indices(len(self))
            idx = slice(start + self._start, stop + self._start)

            return ParquetFile(
                self.path,
                _start=idx.start,
                _take=max(0, idx.stop - idx.start),
                _row_group_sizes=self._row_group_sizes,
            )

        return first(self[idx:])

    def __len__(self):
        if self._take is not None:
            return self._take

        # One would think that pq.ParquetFile had a nicer way to get its length
        return max(0, self.reader.metadata.num_rows - self._start)
