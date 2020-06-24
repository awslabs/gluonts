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

# Standard library imports
from typing import List, Optional, Tuple, Union

import mxnet as mx
import numpy as np
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import get_mxnet_context, validated
from gluonts.dataset.common import Dataset
from gluonts.model.common import Tensor

from .representation import Representation


class ConstNormalization(Representation):
    """
    A class representing a normalization by a constant.

    Parameters
    ----------
    const
        Constant by which to normalize.
    """

    @validated()
    def __init__(
        self, const: int, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.const = const

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
        data = data / self.const
        return data, scale, rep_params

    def post_transform(
        self, F, samples: Tensor, scale: Tensor, rep_params: List[Tensor]
    ) -> Tensor:
        samples = samples * F.full(1, self.const)
        return samples
