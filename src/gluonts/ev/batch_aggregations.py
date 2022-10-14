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

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from gluonts.ev.helpers import axis_is_zero_or_none


@dataclass
class BatchAggregation:
    results: list = field(default_factory=list)

    def step(self) -> None:
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        self.results = []

@dataclass
class Sum(BatchAggregation):
    axis: Optional[int] = None

    def step(self, values) -> None:
        self.results.append(np.sum(values, axis=self.axis))

    def get(self) -> np.ndarray:
        if axis_is_zero_or_none(self.axis):
            return np.sum(self.results, axis=self.axis)

        return np.concatenate(self.results)


@dataclass
class Mean(BatchAggregation):
    axis: Optional[int] = None
    n: int = 0

    def step(self, values) -> None:
        self.results.append(np.sum(values, axis=self.axis))

        if self.axis is None:
            self.n += values.size
        else:
            self.n += values.shape[self.axis]

    def get(self) -> np.ndarray:
        if axis_is_zero_or_none(self.axis):
            return np.sum(self.results, axis=self.axis) / self.n

        return np.concatenate(self.results) / self.n

    def reset(self) -> None:
        super().reset()
        self.n = 0
