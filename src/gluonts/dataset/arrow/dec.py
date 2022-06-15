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

from dataclasses import dataclass
from functools import singledispatch
from typing import Dict

import numpy as np
import pyarrow as pa


@singledispatch
def _arrow_to_py(scalar):
    """Convert arrow scalar value to python value."""

    raise NotImplementedError(scalar, scalar.__class__)


@_arrow_to_py.register(pa.Scalar)
def _arrow_to_py_scalar(scalar: pa.Scalar):
    return scalar.as_py()


@_arrow_to_py.register(pa.ListScalar)
def _arrow_to_py_list_scalar(scalar: pa.ListScalar):
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
