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

from pathlib import Path
from typing import Union, Dict, List, Any, cast, Optional, Tuple
import re

import numpy as np
import pandas as pd
import pyarrow as pa

_ARROW_NUMPY_SHAPE_SUFFIX = "._np_shape"


class ArrowWriter:
    def __init__(
        self,
        path: Union[str, Path],
        chunk_size: int = 100,
        int_dtype=np.int32,
        float_dtype=np.float32,
    ):
        """
        Write records to an arrow table. The first record is used to infer the schema.
        Lists are converted to numpy int or float arrays if possible.
        Strings of the form 'yyyy-mm-dd ...' are converted to timestamps if possible.

        Note:
        Numpy arrays with dimension > 1 are stored in two columns one with a 1d flat data
        an additional column called "<array_name>._np_shape" that stores the array shape.
        One could also do this via pyarrow/pandas extension types, but it seemed overly
        complicated for this use case.
        """
        self.path: Path = Path(path)
        if not str(path).endswith(".arrow"):
            raise RuntimeError("file should end with '.arrow'")
        self.chunk_size = chunk_size
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self._cur_cols: Dict[str, List] = {}
        self._cur_n = 0
        self._numpy_shape_cols: Dict[str, List] = {}
        self._cols_to_numpy_dtype: Dict[str, Any] = {}
        self._timestamp_cols: List[str] = []
        self._writer = None
        self._schema = None
        self._first_record = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _infer_types(self, record):
        for k, v in record.items():
            self._cur_cols[k] = []
            if isinstance(v, list):
                try:
                    tmp = np.asarray(v)
                    dtype_kind = tmp.dtype.kind
                except:
                    dtype_kind = None
                if dtype_kind == "f":
                    self._cols_to_numpy_dtype[k] = self.float_dtype
                    v = tmp
                if dtype_kind == "i":
                    self._cols_to_numpy_dtype[k] = self.int_dtype
                    v = tmp
            if isinstance(v, np.ndarray) and v.ndim > 1:
                self._numpy_shape_cols[k] = []
            if isinstance(v, str) and re.match(r"^\s*\d{4}-\d{2}-\d{2}", v):
                try:
                    v = pd.Timestamp(v)
                    self._timestamp_cols.append(k)
                except:
                    pass
        self._first_record = False

    def write_record(self, d: Dict[str, Any]):
        if self._first_record:
            self._infer_types(d)

        for k, v in d.items():
            conv_dtype = self._cols_to_numpy_dtype.get(k)
            if conv_dtype:
                v = np.asarray(v).astype(conv_dtype)
            if k in self._timestamp_cols:
                v = pd.Timestamp(v)
            if k in self._numpy_shape_cols:
                self._numpy_shape_cols[k].append(list(v.shape))
                self._cur_cols[k].append(v.flatten())
            else:
                self._cur_cols[k].append(v)
        self._cur_n += 1
        if self._cur_n >= self.chunk_size:
            self._write_chunk()

    def _write_chunk(self):
        assert self._cur_n > 0
        recs = {}
        for k, col in self._cur_cols.items():
            recs[k] = col
            self._cur_cols[k] = []
        for k, col in self._numpy_shape_cols.items():
            recs[f"{k}{_ARROW_NUMPY_SHAPE_SUFFIX}"] = col
            self._numpy_shape_cols[k] = []
        self._cur_n = 0
        df = pd.DataFrame(recs)
        table = pa.Table.from_pandas(df)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pa.RecordBatchStreamWriter(
                pa.OSFile(str(self.path), "wb"), self._schema
            )
        chunks = table.to_batches(max_chunksize=self.chunk_size)
        for chunk in chunks:
            self._writer.write_batch(chunk)

    def close(self):
        if self._cur_n:
            self._write_chunk()
        if self._writer:
            self._writer.close()


class ArrowReader:
    def __init__(self, table: pa.Table, chunk_size: int = 100):
        self.table = table
        self.chunk_size = chunk_size
        self._np_shape_cols: Optional[List[Tuple[str, str]]] = None

    def iter_slice(
        self, start: Optional[int] = None, length: Optional[int] = None
    ):
        start = start if start is not None else 0
        length = length if length is not None else len(self.table) - start
        if start > 0 or length < len(self.table):
            table = self.table.slice(start, length)
        else:
            table = self.table

        for chunk in table.to_batches(self.chunk_size):
            chunk = cast(pa.RecordBatch, chunk)
            df: pd.DataFrame = chunk.to_pandas()
            records = df.to_dict(orient="records")
            if self._np_shape_cols is None:
                rec0 = records[0]
                self._np_shape_cols = [
                    (k[: -len(_ARROW_NUMPY_SHAPE_SUFFIX)], k)
                    for k in rec0.keys()
                    if k.endswith(_ARROW_NUMPY_SHAPE_SUFFIX)
                ]
            for rec in records:
                for k, k_shape in self._np_shape_cols:
                    rec[k] = rec[k].reshape(*rec[k_shape])
                    del rec[k_shape]
                yield rec

    def __iter__(self):
        yield from self.iter_slice()
