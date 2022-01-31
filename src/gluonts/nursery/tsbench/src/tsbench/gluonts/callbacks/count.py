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
from mxnet.gluon import nn
from .base import Callback


class ParameterCountCallback(Callback):  # type: ignore
    """
    This callback allows counting model parameters during training.

    Attributes:
        num_parameters: The number of parameters of the model. This attribute should only be
            accessed after training.
    """

    def __init__(self) -> None:
        self.num_parameters = 0

    def on_network_initialization_end(self, network: nn.HybridBlock) -> None:
        self.num_parameters = sum(
            np.prod(p.shape) for p in network.collect_params().values()
        )
