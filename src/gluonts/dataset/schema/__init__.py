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

from dataclasses import dataclass, field, MISSING
from typing import Any, Dict, Type as typingType, Union

from .translate import Translator
from .types import Type, Array, Default, Period

__all__ = [
    # this module
    "Schema",
    # types
    "Type",
    "Array",
    "Default",
    "Period",
    # translate
    "Translator",
]


TypeType = Union[typingType, Type]


@dataclass
class Schema:
    fields: Dict[str, TypeType] = field(default_factory=dict)
    default_fields: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.default_fields = {}

        for name, ty in list(self.fields.items()):
            if isinstance(ty, Default):
                self.default_fields[name] = self.fields.pop(name).value

    def apply(self, entry) -> dict:
        result = {
            field_name: ty(entry[field_name])
            for field_name, ty in self.fields.items()
        }

        for name, default in self.default_fields.items():
            result[name] = default
        return result

    def add(self, name: str, ty: TypeType):
        self.fields[name] = ty

        return self

    def add_if(
        self, condition: bool, name: str, ty: TypeType, default=MISSING
    ):
        if not condition:
            if default is not MISSING:
                self.default_fields[name] = ty(default)
            return self

        return self.add(name, ty)
