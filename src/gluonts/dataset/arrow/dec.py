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
from typing import List, Tuple

import numpy as np


@dataclass
class ArrowDecoder:
    reshape_columns: List[Tuple[str, str]]

    @classmethod
    def from_schema(cls, schema):
        return cls(
            [
                (column.name[: -len("._np_shape")], column.name)
                for column in schema
                if column.name.endswith("._np_shape")
            ]
        )

    def decode(self, batch, row_number: int):
        yield from self.decode_batch(batch.slice(row_number, row_number + 1))

    def decode_batch(self, batch):
        for row in batch.to_pandas().to_dict("records"):
            for column_name, shape_column in self.reshape_columns:
                row[column_name] = row[column_name].reshape(
                    row.pop(shape_column)
                )

            for name, value in row.items():
                if type(value) == np.ndarray and value.dtype == object:
                    row[name] = np.stack(value)

            yield row
