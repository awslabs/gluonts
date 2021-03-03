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

from typing import List, Optional, Tuple

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .representation import Representation


class DimExpansion(Representation):
    """
    A class representing a dimension expansion operation along a specified axis.

    Parameters
    ----------
    axis
        Axis on which to expand the tensor.
        (default: -1)
    """

    @validated()
    def __init__(self, axis: int = -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        observed_indicator: Tensor,
        scale: Optional[Tensor],
        rep_params: List[Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        data = F.expand_dims(data, axis=self.axis)
        return data, scale, rep_params
