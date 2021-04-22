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

import re
from pathlib import Path
from typing import (
    Union,
    Dict,
    List,
    Any,
    cast,
    Optional,
    Tuple,
    Iterator,
    Iterable,
)

import numpy as np
import pandas as pd
import pyarrow as pa
from gluonts.dataset import util
from gluonts.dataset.util import get_bounds_for_mp_data_loading
from gluonts.gluonts_tqdm import tqdm

from .common import Dataset, DataEntry, ProcessDataEntry, SourceContext

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
        if not self.path.suffix == ".arrow":
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

    def iter_slice(self, start: int = 0, length: Optional[int] = None):
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


class ArrowDataset(Dataset):
    _SOURCE_COL_NAME = "_GLUONTS_SOURCE"
    _ROW_COL_NAME = "_GLUONTS_ROW"

    def __init__(
        self,
        table: pa.Table,
        freq: str,
        chunk_size: int = 100,
    ):
        """

        An on disk Dataset based on pyarrow tables that is faster than json for large datasets or
        datasets with long time series.

        Use `ArrowDataset.load_files` to load the dataset from disk and
        `ArrowDataset.write_table_from_records` to save recors in arrow format.
        """
        self.table = table
        self.chunk_size = chunk_size
        self.freq = freq
        self._process = ProcessDataEntry(self.freq, one_dim_target=True)
        self.reader = ArrowReader(self.table, chunk_size=self.chunk_size)

    @classmethod
    def load_files(
        cls,
        paths: Union[str, Path, List[str], List[Path]],
        freq: str,
        chunk_size: int = 100,
    ):
        paths = util.resolve_paths(paths)
        files = util.find_files(paths, lambda p: p.suffix == ".arrow")
        assert files, f"Could not find any arrow files in paths: {paths}"
        return MemmapArrowDataset(files, freq, chunk_size)

    def __iter__(self) -> Iterator[DataEntry]:
        bounds = get_bounds_for_mp_data_loading(len(self))
        for row_number, rec in enumerate(
            self.reader.iter_slice(bounds.lower, bounds.upper),
            start=bounds.lower,
        ):
            if self._SOURCE_COL_NAME in rec:
                rec["source"] = SourceContext(
                    source=rec[self._SOURCE_COL_NAME],
                    row=rec[self._ROW_COL_NAME],
                )
                del rec[self._SOURCE_COL_NAME]
                del rec[self._ROW_COL_NAME]
            else:
                rec["source"] = SourceContext(
                    source="arrow.Table", row=row_number
                )
            rec = self._process(rec)
            yield rec

    def __len__(self):
        return len(self.table)

    @classmethod
    def write_table_from_records(
        cls,
        it: Iterable[DataEntry],
        file_path: Union[str, Path],
        chunk_size=100,
    ) -> None:
        """
        Write time series records as an arrow dataset to the file_path.
        """
        file_path = Path(file_path)
        msg = f"Converting to arrow dataset: {file_path}"
        length: Optional[int] = None
        try:
            length = len(it)  # type: ignore
        except:
            length = None
        with ArrowWriter(file_path, chunk_size=chunk_size) as writer:
            for rec in tqdm(it, total=length, desc=msg):
                rec = rec.copy()
                float_np_fields = [
                    "target",
                    "feat_static_real",
                    "feat_dynamic_real",
                ]
                for field in float_np_fields:
                    if field in rec:
                        rec[field] = np.asarray(rec[field], dtype=np.float32)
                int_np_fields = ["feat_static_cat", "feat_dynamic_cat"]
                for field in int_np_fields:
                    if field in rec:
                        rec[field] = np.asarray(rec[field], dtype=np.int32)
                if "start" in rec:
                    rec["start"] = pd.Timestamp(rec["start"])
                if "source" in rec:
                    del rec["source"]
                writer.write_record(rec)


class MemmapArrowDataset(ArrowDataset):
    """
    Arrow dataset using memory mapped files that closes the files when the object is deleted.
    """

    def __init__(self, files: List[Path], freq: str, chunk_size: int):
        self.files = files
        self.mmaps = [pa.memory_map(str(p)) for p in files]
        tables = []
        for file_path, mm in zip(files, self.mmaps):
            t = pa.ipc.open_stream(mm).read_all()
            source_col = pa.repeat(str(file_path), len(t)).dictionary_encode()
            t = t.append_column(self._SOURCE_COL_NAME, source_col)
            row_col = pa.array(np.arange(len(t)))
            t = t.append_column(self._ROW_COL_NAME, row_col)
            tables.append(t)
        if len(tables) > 1:
            table = pa.concat_tables(tables)
        else:
            table = tables[0]
        super().__init__(table, freq=freq, chunk_size=chunk_size)

    def __del__(self):
        self.close()

    def close(self):
        for mm in self.mmaps:
            mm.close()
