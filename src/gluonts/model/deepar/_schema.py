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

from asyncore import file_dispatcher
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Generic, TypeVar, Type

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.tseries.frequencies import to_offset

from gluonts.exceptions import GluonTSDataError


T = TypeVar("T")


class FieldType(Generic[T]):
    def __call__(self, v: Any) -> T:
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def is_compatible(self, v: Any) -> bool:
        raise NotImplementedError()


# convert functions
def ct_to_tc(xs):
    return xs.transpose()


def tc_to_ct(xs):
    return xs.transpose()


def t_to_tc(xs):
    return xs.reshape((-1, 1))


def t_to_ct(xs):
    return xs.reshape((1, -1))


conv_map = {
    ("TC", "CT"): tc_to_ct,
    ("CT", "TC"): tc_to_ct,
    ("T", "CT"): t_to_ct,
    ("T", "TC"): t_to_tc,
}


@dataclass
class NumpyArrayField(FieldType[np.ndarray]):
    dtype: Type = np.float32
    ndim: Optional[int] = None
    src_layout: Optional[str] = "T"
    target_layout: Optional[str] = "T"

    def __call__(self, value: Any) -> np.ndarray:
        if isinstance(value, pa.Array):
            value = value.to_numpy()
        else:
            value = np.asarray(value, dtype=self.dtype)
        if self.ndim is not None and self.ndim != value.ndim:
            raise GluonTSDataError(
                f"expected array with dimension {self.ndim}, "
                "but got {value.ndim}."
            )

        # layout tranformation
        if self.src_layout != self.target_layout:
            conv_func = conv_map[(self.src_layout, self.target_layout)]
            value = conv_func(value)

        return value

    def is_compatible(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple, np.ndarray, pa.Array)):
            return False

        try:
            if isinstance(value, pa.Array):
                x = value.to_numpy()
            else:
                x = np.asarray(value)
        except Exception:
            return False

        # int types
        if self.dtype in [np.int32, np.int64]:
            if x.dtype.kind != "i":
                return False
            return self.ndim is None or x.ndim == self.ndim

        # float types
        try:
            x = np.asarray(value, dtype=self.dtype)
        except Exception:
            return False
        return self.ndim is None or x.ndim == self.ndim


Freq = Union[str, pd.DateOffset]


class PandasPeriodField(FieldType[pd.Period]):
    def __init__(
        self,
        freq: Freq,
    ) -> None:
        self.freq = to_offset(freq) if isinstance(freq, str) else freq

    def __eq__(self, other):
        if not isinstance(other, PandasPeriodField):
            return False
        return self.freq == other.freq

    def __repr__(self):
        return f"PandasPeriodField(freq={self.freq!r})"

    def __call__(self, value: Any) -> pd.Period:
        return pd.Period(value, freq=self.freq)

    def is_compatible(self, v: Any) -> bool:
        if not isinstance(v, (str, pd.Period)):
            return False
        try:
            self(v)
        except Exception:
            return False
        return True


@dataclass
class Schema:
    fields: Dict[str, FieldType]
    default_values: Dict[str, Any]

    def __call__(
        self,
        d: Dict[str, Any],
        inplace: bool,
        src_layout: Dict[str, str],
        name_mapping: Dict[str, str],
    ) -> Dict[str, FieldType]:
        """
        inplace
            True: applies the schema to the input dictionary.
                  The dictionary is updated in place.
            False: return a new data dictionary if False.

        name_mapping: mapping from target name to src name
        """
        if inplace:
            out: Dict[str, Any] = d
        else:
            out = {}

        for field_name, field_type in self.fields.items():
            try:
                src_fields = name_mapping[field_name]
                if isinstance(src_fields, list):
                    value = []
                    for field in src_fields:
                        value.append(d[field])
                    self.fields[field_name].src_layout = "CT"
                else:
                    value = d[src_fields]
                    self.fields[field_name].src_layout = src_layout[src_fields]
            except KeyError:
                if field_name in self.default_values.keys():
                    value = self.default_values[field_name]
                else:
                    raise GluonTSDataError(
                        f"field {field_name} does not occur in the data"
                    )

            try:
                out[field_name] = field_type(value)
            except Exception as e:
                raise GluonTSDataError(
                    f"Error when processing field {field_name} using "
                    "{field_type}"
                ) from e
        return out

    @staticmethod
    def infer(entry: Dict[str, Any]) -> "Schema":
        """
        Infers the schema from the passed data entry
        """
        if not "start" in entry:
            raise GluonTSDataError(
                "start not provided and could not be found in data entry"
            )

        if "freq" in entry:
            found_freq = to_offset(entry["freq"])
        else:
            # infer freq from start time
            if isinstance(entry["start"], (pd.Period, pd.Timestamp)):
                found_freq = entry["start"].freq
            else:
                raise GluonTSDataError(
                    "freq not provided and could not be inferred from start"
                )

        candidate_types = [
            PandasPeriodField(freq=found_freq),
            NumpyArrayField(dtype=np.int32),
            NumpyArrayField(dtype=np.float32),
        ]

        fields = {}

        for field_name in entry:
            value = entry[field_name]
            inferred_fields = [
                ct for ct in candidate_types if ct.is_compatible(value)
            ]
            fields[field_name] = inferred_fields[0]
        return Schema(fields, {})
