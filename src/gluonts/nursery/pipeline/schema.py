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
from typing import Any, Dict, Type, Union

import numpy as np


@dataclass
class Schema:
    fields: Dict[str, Type] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.fields[key]

    def get(self, key, default):
        return self.fields.get(key, default)

    def __contains__(self, key):
        return key in self.fields

    def copy(self):
        return Schema(self.fields.copy())

    def add(self, name: str, ty: Type, force=False) -> "Schema":
        if name in self and not force:
            raise KeyError(f"Can't override existing field {name!r}.")

        clone = self.copy()
        clone.fields[name] = ty
        return clone

    def union(self, name: str, ty: Type) -> "Schema":
        clone = self.copy()
        clone.fields[name] = Union[self.fields.get(name, ty), ty]
        return clone

    def remove(self, name: str, ignore_missing=False) -> "Schema":
        if name not in self and not ignore_missing:
            raise KeyError(f"Can't remove missing field {name!r}.")

        clone = self.copy()
        clone.fields.pop(name, None)
        return clone

    def pop(self, name: str) -> "Schema":
        if name not in self:
            raise KeyError(f"Can't pop missing field {name!r}.")

        clone = self.copy()
        ty = clone.fields.pop(name)
        return clone, ty

    def validate(self, entry: dict):
        return {name: ty(entry[name]) for name, ty in self.fields.items()}

    @classmethod
    def infer(cls, data):
        return cls({name: infer_type(value) for name, value in data.items()})

    @classmethod
    def dryrun(cls, pipeline):
        return pipeline.apply_schema(SchemaTracker())


@dataclass
class SchemaTracker:
    output_schema: Dict[str, Type] = field(default_factory=Schema)
    input_schema: Dict[str, Type] = field(default_factory=Schema)

    def _with_missing(self, key, Ty=Any):
        if key in self.output_schema:
            return self, self.output_schema[key]

        Ty_ = self.input_schema.get(key, Ty)
        combined = Union[Ty, Ty_]
        clone = self.copy()

        clone.input_schema.fields[key] = combined
        return clone, combined

    def copy(self):
        return SchemaTracker(
            self.output_schema.copy(), self.input_schema.copy()
        )

    def add(self, name: str, ty: Type, force=False) -> "Schema":
        if name in self.output_schema and not force:
            raise KeyError(f"Can't override existing field {name!r}.")

        elif name in self.input_schema and not force:
            raise KeyError(f"Overwrite shadowed field {name!r}.")

        clone = self.copy()
        clone.output_schema.fields[name] = ty
        return clone

    def union(self, name: str, ty: Type) -> "Schema":
        clone = self.copy()
        clone.output_schema.fields[name] = Union[
            self.output_schema.get(name, ty), ty
        ]
        return clone

    def remove(self, name: str, ignore_missing=False) -> "Schema":
        schema, _ty = self._with_missing(name)
        schema.output_schema.fields.pop(name, None)
        return schema

    def pop(self, name: str) -> "Schema":
        schema, ty = self._with_missing(name)

        schema.output_schema.fields.pop(name, None)
        return schema, ty


@singledispatch
def infer_type(val: Any):
    return type(val)
