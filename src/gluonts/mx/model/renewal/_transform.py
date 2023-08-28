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
from gluonts.transform import SimpleTransformation


class AddAxisLength(SimpleTransformation):
    """
    Add the "length" of an array along the specified axis.

    For example, if target field in the input DataEntry has a shape
    of (32, 24, 3), the axis can be specified as 1 to return 24 for
    the output field. If the target field is a list instead, its
    length will be added.

    Parameters
    ----------
    target_field
        Field with target values (array) of time series
    axis
        Axis to output lengths (dimensions) on. Default 0.
    output_field
        Name of newly created field
    """

    @validated()
    def __init__(
        self,
        target_field: str = FieldName.TARGET,
        axis: int = 0,
        output_field: str = "valid_length",
    ) -> None:
        self.target_field = target_field
        self.axis = axis
        self.output_field = output_field

    def transform(self, data: DataEntry) -> DataEntry:
        target = data[self.target_field]
        data[self.output_field] = np.array(
            [
                len(target)
                if isinstance(target, list)
                else target.shape[self.axis]
            ]
        )
        return data
