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

__all__ = [
    # types
    "Type",
    "Default",
    "Array",
    "Period",
    # this module
    "Schema",
]

from dataclasses import dataclass, field
from typing import Any, Dict

from .types import Type, Default, Array, Period


@dataclass
class SchemaBuilder:
    def add(self, field):
        pass


@dataclass
class Schema:
    fields: Dict[str, Type]
    default_fields: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.default_fields = {}

        for name, ty in list(self.fields.items()):
            if isinstance(ty, Default):
                self.default_fields[name] = self.fields.pop(name).value

    def apply(self, entry):
        result = {
            field_name: ty.apply(entry[field_name])
            for field_name, ty in self.fields.items()
        }

        for name, default in self.default_fields.items():
            result[name] = default
        return result
