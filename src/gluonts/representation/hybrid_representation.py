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


class HybridRepresentation(Representation):
    """
        A class representing a hybrid approach of combining multiple representations into a single representation.
        Representations will be combined by concatenating them at on dim=1.

        Parameters
        ----------
        representations
            A list of representations. Elements must be of type Representation.
    """

    @validated()
    def __init__(self, representations: List, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representations = representations
        for representation in self.representations:
            self.register_child(representation)

    def initialize_from_dataset(self, input_dataset: Dataset):
        for representation in self.representations:
            representation.initialize_from_dataset(input_dataset)

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        observed_indicator: Optional[Tensor],
        scale: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        representation_data_agg = None

        for representation in self.representations:
            representation_data, _ = representation(
                data, observed_indicator, scale
            )
            if self.is_output:
                representation_data = representation_data.expand_dims(
                    -1
                ).swapaxes(1, 2)

            if representation_data_agg is None:
                representation_data_agg = representation_data
            else:
                representation_data_agg = F.concat(
                    representation_data_agg, representation_data, dim=1
                )

        if scale is None:
            scale = F.expand_dims(
                F.sum(data, axis=-1) / F.sum(observed_indicator, axis=-1), -1
            )

        return representation_data_agg, scale

    def post_transform(self, F, samples: Tensor):
        raise NotImplementedError
