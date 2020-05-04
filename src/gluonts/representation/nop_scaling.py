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

from .representation import Representation

# Standard library imports
from typing import Tuple, Optional, List

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class NOPScaling(Representation):
    """
    A class representing NOP (no operation) scaler.
    As the name suggests, this scaler will not alter its inputs.
    """

    @validated()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        data = F.cast(data, dtype="float32")

        if scale is None:
            scale = F.ones_like(data)
            scale = scale.expand_dims(axis=1)

        return data, scale, []
