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
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar, Type

import numpy as np
import pandas as pd
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


@dataclass
class NumpyArrayField(FieldType[np.ndarray]):
    dtype: Type = np.float32
    ndim: Optional[int] = None

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


Freq = Union[str, pd.DateOffset]


class PandasPeriodField(FieldType[pd.Period]):
    def __init__(self, freq: Freq) -> None:
        self.freq = to_offset(freq)

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
class PyArrayField:
    shape: List
    dtype: Type

    def __call__(self, value: Any) -> List:
        value = list(value)
        return value


@dataclass
class FieldWithDefault:
    field_type: Any
    default_value: Any

    def __call__(self, value: Any) -> Any:
        return self.field_type(value)


@dataclass
class Schema:
    fields: Dict[str, Any]

    def __call__(
        self, d: Dict[str, Any], inplace: bool = True
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
                value = d[field_name]
            except KeyError:
                if isinstance(field_type, FieldWithDefault):
                    value = field_type.default_value
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

    @staticmethod
    def infer(entry: Dict[str, Any]):
        """
        Infers the schema from the passed data entry
        """
        out: Dict[str, Any] = {}
        for field in entry:
            if not isinstance(entry[field], list):
                out[field] = type(entry[field])
            else:
                first_dim = len(entry[field])
                if isinstance(entry[field][0], list):
                    second_dim = len(entry[field][0])
                    array_shape = [first_dim, second_dim]
                    dtype = type(entry[field][0][0])
                else:
                    array_shape = [first_dim,]
                    dtype = type(entry[field][0])
                out[field] = PyArrayField(array_shape, dtype)
        return Schema(out)
