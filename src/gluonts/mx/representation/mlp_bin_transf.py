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


class MLPBinningTransformation(Representation):
    """
    A class representing an MLP which can learn an appropriate binning, effectively learning an embedding on top on 
    non-discrete data.

    Parameters
    ----------
    num_bins
        Binning resolution.
    embedding_size
        The desired MLP output size. By default, the following heuristic is used:
        https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        (default: round(num_bins**(1/4)))
    """

    @validated()
    def __init__(
        self,
        num_bins: int,
        embedding_size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_bins = num_bins

        if embedding_size is None:
            self.embedding_size = round(self.num_bins ** (1 / 4))
        else:
            self.embedding_size = embedding_size

        self.mlp = nn.HybridSequential()
        self.mlp.add(
            nn.Dense(units=self.num_bins, activation="relu", flatten=False)
        )
        self.mlp.add(nn.Dense(units=self.embedding_size, flatten=False))

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
        data = F.expand_dims(data, axis=-1)
        data = self.mlp(data)
        return data, scale, rep_params
