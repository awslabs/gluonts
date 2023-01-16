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

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    MapTransformation,
    target_transformation_length,
)


class BroadcastTo(MapTransformation):
    @validated()
    def __init__(
        self,
        field: str,
        ext_length: int = 0,
        target_field: str = FieldName.TARGET,
    ) -> None:
        self.field = field
        self.ext_length = ext_length
        self.target_field = target_field

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.ext_length, is_train
        )
        data[self.field] = np.broadcast_to(
            data[self.field],
            (data[self.field].shape[:-1] + (length,)),
        )
        return data
