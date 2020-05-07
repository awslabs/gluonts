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
import numpy as np
from typing import Tuple, Optional, List
import mxnet as mx

# First-party imports
from gluonts.core.component import validated, get_mxnet_context
from gluonts.model.common import Tensor
from gluonts.dataset.common import Dataset


class RepresentationChain(Representation):
    """
    A class representing a hybrid approach of combining multiple representations into a single representation.
    Representations will be combined by concatenating them on dim=-1.

    Parameters
    ----------
    representations
        A list of representations. Elements must be of type Representation.
    """

    @validated()
    def __init__(self, chain: List, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain = chain
        for representation in self.chain:
            self.register_child(representation)

    def initialize_from_dataset(
        self, input_dataset: Dataset, ctx: mx.Context = get_mxnet_context()
    ):
        for representation in self.chain:
            representation.initialize_from_dataset(input_dataset, ctx)

    def initialize_from_array(
        self, input_array: np.ndarray, ctx: mx.Context = get_mxnet_context()
    ):
        for representation in self.chain:
            representation.initialize_from_array(input_array, ctx)

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
        for representation in self.chain:
            data, scale, rep_params = representation(
                data, observed_indicator, scale, rep_params,
            )
        return data, scale, rep_params

    def post_transform(
        self, F, samples: Tensor, scale: Tensor, rep_params: List[Tensor]
    ) -> Tensor:
        for representation in self.chain[::-1]:
            samples = representation.post_transform(
                F, samples, scale, rep_params,
            )
        return samples
