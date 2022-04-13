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

from typing import List
import numpy as np
import numpy.typing as npt
from tsbench.config import Config
from ._base import Surrogate, T
from ._factory import register_ensemble_surrogate, register_surrogate


@register_surrogate("random")
@register_ensemble_surrogate("random")
class RandomSurrogate(Surrogate[T]):
    """
    The random surrogate simply predicts random performance metrics to act as a
    baseline.
    """

    num_outputs_: int

    def _fit(self, X: List[Config[T]], y: npt.NDArray[np.float32]) -> None:
        self.num_outputs_ = y.shape[1]

    def _predict(self, X: List[Config[T]]) -> npt.NDArray[np.float32]:
        return np.random.rand(len(X), self.num_outputs_).astype(np.float32)
