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

import abc
from typing import Any, Dict, List

from gluonts.dataset.common import DataEntry
from ._base import SimpleTransformation


class Validation(SimpleTransformation):
    """A Transformation that doesn't change the data, but just checks whether a
    given entry is valid.
    """

    @abc.abstractmethod
    def check(self, data: DataEntry) -> None:
        pass

    def transform(self, data: DataEntry) -> DataEntry:
        self.check(data)
        return data


class ContainsField(Validation):
    def __init__(self, *field_names: List[str]) -> None:
        self.field_names = set(field_names)

    def check(self, data: DataEntry) -> None:
        missing = self.field_names - set(data)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Entry is missing fields: {fields}.")


class HasType(Validation):
    def __init__(self, **types: Dict[Any, type]) -> None:
        self.types = types

    def check(self, data: DataEntry) -> None:
        for field, expected_type in self.types.items():
            if not isinstance(data[field], expected_type):
                raise ValueError(
                    f"Entry {field} was expected to be of instance "
                    f"{expected_type}, but found type {type(data[field])}."
                )
