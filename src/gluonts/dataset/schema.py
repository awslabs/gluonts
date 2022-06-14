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
from typing import Any, Dict, Optional, Union, Generic, TypeVar, Type

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.tseries.frequencies import to_offset

from gluonts.exceptions import GluonTSDataError


@dataclass
class PyArray:
    array_shape: list
    dtype: Type

    def __str__(self):
        return "array(shape=%s, dtype=%s)" % (self.array_shape, self.dtype)


T = TypeVar("T")


class FieldType(Generic[T]):
    def __call__(self, v: Any) -> T:
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def is_compatible(self, v: Any) -> bool:
        raise NotImplementedError()


@dataclass
class NumpyArrayField(FieldType[np.ndarray]):
    dtype: Type = np.float32
    ndim: Optional[int] = None

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
    targert_layout: Dict[str, str] = {}

    # convert functions
    def ct_to_tc(xs):
        return xs.transpose()

    def tc_to_ct(xs):
        return xs.transpose()

    def t_to_tc(xs):
        return xs.reshape((-1, 1))

    def t_to_ct(xs):
        return xs.reshape((1, -1))

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

        # map for layout conversion
        conv_map = {
            ("TC", "CT"): Schema.tc_to_ct,
            ("CT", "TC"): Schema.tc_to_ct,
            ("T", "CT"): Schema.t_to_ct,
            ("T", "TC"): Schema.t_to_tc,
        }

        for field_name, field_type in self.fields.items():
            try:
                src_fields = name_mapping[field_name]
            except KeyError:
                if field_name in self.default_values.keys():
                    value = self.default_values[field_name]
                else:
                    raise GluonTSDataError(
                        f"field {field_name} does not occur in the data"
                    )

            # multi-src-field for target field
            if isinstance(src_fields, list):
                value = []
                for field in src_fields:
                    value.append(d[field])
                src_layout[field_name] = "CT"
            else:
                value = d[src_fields]
                src_layout[field_name] = src_layout.pop(src_fields)

            try:
                # type conversion
                value = field_type(value)

                # layout conversion
                # TODO: check what field should have layout conversion
                layout_s = src_layout[field_name]
                layout_t = self.targert_layout[field_name]
                if layout_s != layout_t:
                    conv_func = conv_map[(layout_s, layout_t)]
                    value = conv_func(value)

                out[field_name] = value

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
        # get raw data type
        raw_type_mapping: Dict[str, Any] = {}
        for field in entry:
            if not isinstance(entry[field], list):
                raw_type_mapping[field] = type(entry[field])
            else:
                first_dim = len(entry[field])
                second_dim = len(entry[field][0])
                if second_dim > 0:
                    array_shape = [first_dim, second_dim]
                    dtype = type(entry[field][0][0])
                else:
                    array_shape = [first_dim]
                    dtype = type(entry[field][0])
                raw_type_mapping[field] = PyArray(array_shape, dtype)

        # TODO: map raw type to candidate types

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
