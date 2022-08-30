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

from collections import Counter
from typing import Any, Dict, List

from gluonts.core import serde
from gluonts.dataset.common import DataEntry

from ._base import MapTransformation, SimpleTransformation


@serde.dataclass
class RenameFields(SimpleTransformation):
    """
    Rename fields using a mapping, if source field present.

    Parameters
    ----------
    mapping
        Name mapping `input_name -> output_name`
    """

    mapping: Dict[str, str]

    def __post_init_post_parse__(self):
        values_count = Counter(self.mapping.values())
        for new_key, count in values_count.items():
            assert count == 1, f"Mapped key {new_key} occurs multiple time"

    def transform(self, data: DataEntry):
        for key, new_key in self.mapping.items():
            if key in data:
                # no implicit overriding
                assert new_key not in data
                data[new_key] = data[key]
                del data[key]
        return data


@serde.dataclass
class RemoveFields(SimpleTransformation):
    """
    " Remove field names if present.

    Parameters
    ----------
    field_names
        List of names of the fields that will be removed
    """

    field_names: List[str]

    def transform(self, data: DataEntry) -> DataEntry:
        for k in self.field_names:
            data.pop(k, None)
        return data


@serde.dataclass
class SetField(SimpleTransformation):
    """
    Sets a field in the dictionary with the given value.

    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    output_field: str
    value: Any

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.output_field] = self.value
        return data


@serde.dataclass
class SetFieldIfNotPresent(SimpleTransformation):
    """
    Sets a field in the dictionary with the given value, in case it does not
    exist already.

    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    field: str
    value: Any

    def transform(self, data: DataEntry) -> DataEntry:
        if self.field not in data.keys():
            data[self.field] = self.value
        return data


@serde.dataclass
class SelectFields(MapTransformation):
    """
    Only keep the listed fields.

    Parameters
    ----------
    input_fields
        List of fields to keep.
    allow_missing
        If ``True``, skip any missing field. Default: ``False``.
    """

    input_fields: List[str]
    allow_missing: bool = False

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if self.allow_missing:
            return {f: data[f] for f in self.input_fields if f in data}
        return {f: data[f] for f in self.input_fields}
