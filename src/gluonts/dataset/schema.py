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
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from gluonts.exceptions import GluonTSDataError


@dataclass
class FieldType:
    raw_type: Type
    args: Optional[dict] = None

    def __call__(self, v: Any):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def is_compatible(self, v: Any) -> bool:
        raise NotImplementedError()


@dataclass
class PlainFieldType(FieldType):
    def __call__(self, v: Any):
        return self.raw_type(v)


@dataclass
class NumpyArrayField(FieldType):
    raw_type: Type = np.ndarray

    def __post_init__(self):
        self.dtype = self.args["dtype"] if "dtype" in self.args else np.float32
        self.ndim = self.args["ndim"] if "ndim" in self.args else None

    def __call__(self, value: Any) -> np.ndarray:
        value = np.asarray(value, dtype=self.dtype)

        if self.ndim is not None and self.ndim != value.ndim:
            raise GluonTSDataError(
                f"expected array with dimension {self.ndim}, "
                "but got {value.ndim}."
            )

        return value

    def is_compatible(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple, np.ndarray)):
            return False

        try:
            x = np.asarray(value)
        except Exception:
            return False

        # int types
        if self.dtype in [np.int32, np.int64, int]:
            # float is not int
            if x.dtype.kind != "i":
                return False
            return self.ndim is None or x.ndim == self.ndim

        # float types
        try:
            # int can be float
            x = np.asarray(value, dtype=self.dtype)
        except Exception:
            return False
        return self.ndim is None or x.ndim == self.ndim


@dataclass
class PandasPeriodField(FieldType):
    raw_type: Type = pd.Period

    def __post_init__(self):
        assert "freq" in self.args
        self.freq = to_offset(self.args["freq"])

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


def get_field_type(raw_type: Type, args: Optional[dict] = None) -> FieldType:
    if raw_type == np.ndarray:
        return NumpyArrayField(raw_type, args)
    if raw_type == pd.Period:
        return PandasPeriodField(raw_type, args)
    return PlainFieldType(raw_type, args)


@dataclass
class Schema:
    fields: Dict[str, FieldType]

    def __call__(
        self, d: Dict[str, Any], inplace: bool = True, name_mapping: Dict = {}
    ) -> Dict[str, Any]:
        """
        inplace
            True: applies the schema to the input dictionary.
                  The dictionary is updated in place.
            False: return a new data dictionary if False.
        """
        if inplace:
            out: Dict[str, Any] = d
        else:
            out = {}

        for field_name, field_type in self.fields.items():
            try:
                if field_name in name_mapping:
                    if isinstance(name_mapping[field_name], list):
                        value = []
                        for sub in name_mapping[field_name]:
                            value.append(d[sub])
                    else:
                        value = d[name_mapping[field_name]]
                else:
                    value = d[field_name]
            except KeyError:
                if field_type.args and "default" in field_type.args:
                    value = field_type.args["default"]
                else:
                    raise GluonTSDataError(
                        f"field {field_name} does not occur in the data"
                    )

            try:
                # type conversion
                out[field_name] = field_type(value)

            except Exception as e:
                raise GluonTSDataError(
                    f"Error when processing field {field_name} using "
                    "{field_type}"
                ) from e
        return out

    @classmethod
    def infer(cls, d: Dict[str, Any]) -> "Schema":
        fields: Dict[str, FieldType] = {}
        for field_name, value in d.items():
            if isinstance(value, np.ndarray):
                fields[field_name] = get_field_type(
                    np.ndarray,
                    {
                        "dtype": value.dtype,
                        "ndim": value.ndim,
                        "shape": value.shape,
                    },
                )
                continue
            if isinstance(value, pd.Period):
                fields[field_name] = get_field_type(
                    pd.Period, args={"freq": value.freq}
                )
                continue
            if isinstance(value, (list, tuple)):
                try:
                    array = np.array(value)
                    fields[field_name] = get_field_type(
                        type(value),
                        {
                            "dtype": array.dtype,
                            "ndim": array.ndim,
                            "shape": array.shape,
                        },
                    )
                except Exception:
                    fields[field_name] = get_field_type(
                        type(value), {"ndim": None}
                    )
                continue

            fields[field_name] = PlainFieldType(raw_type=type(value))

        return Schema(fields)
