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
from typing import List, Set

from gluonts.itertools import batcher, rows_to_columns
from toolz.curried import keyfilter, valmap

import numpy as np
import pandas as pd
import pyarrow as pa


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
    cls, dataset, path, metadata=None, batch_size=1024, flatten_arrays=True
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
        writer = cls.writer()(fobj, schema=schema)
        for batch in chain([first], batches):
            writer.write_batch(batch)

        writer.close()
