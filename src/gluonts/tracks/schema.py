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
from typing import Dict, Type

import numpy as np


@dataclass
class Schema:
    fields: Dict[str, Type] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.fields[key]

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

    def remove(self, name: str, ignore_missing=False) -> "Schema":
        if name not in self and not ignore_missing:
            raise KeyError(f"Can't remove missing field {name!r}.")

        clone = self.copy()
        clone.fields.pop(name, None)
        return clone

    def validate(self, entry: dict):
        return {name: ty(entry[name]) for name, ty in self.fields.items()}
