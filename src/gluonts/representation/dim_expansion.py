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
from gluonts.dataset.common import Dataset


class DimExpansion(Representation):
    """
    A class representing a dimension expansion operation of a given representation along a specified axis.

    Parameters
    ----------
    representation
        The underlying representation.
    axis
        Axis on which to expand the tensor.
        (default: -1)
    """

    @validated()
    def __init__(
        self, representation: Representation, axis: int = -1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.representation = representation
        self.register_child(representation)
        self.axis = axis

    def initialize_from_dataset(self, input_dataset: Dataset):
        self.representation.initialize_from_dataset(input_dataset)

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
        repr_data, scale, rep_params = self.representation(
            data, observed_indicator, scale, rep_params
        )

        repr_data = F.expand_dims(repr_data, axis=self.axis)

        return repr_data, scale, rep_params
