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

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np

from gluonts.ev.helpers import axis_is_zero_or_none


@dataclass
class Aggregation:
    def step(self) -> None:
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Sum(Aggregation):
    axis: Optional[int] = None
    partial_result: Optional[Union[List[np.ndarray], np.ndarray]] = None

    def step(self, values) -> None:
        summed_values = np.sum(values, axis=self.axis)

        if self.partial_result is None:
            if axis_is_zero_or_none(self.axis):
                self.partial_result = np.zeros(shape=np.shape(summed_values))
            else:
                self.partial_result = []

        if axis_is_zero_or_none(self.axis):
            self.partial_result += summed_values
        else:
            self.partial_result.append(summed_values)

    def get(self) -> np.ndarray:
        if axis_is_zero_or_none(self.axis):
            return self.partial_result

        return np.concatenate(self.partial_result)


@dataclass
class Mean(Aggregation):
    axis: Optional[int] = None
    n: int = 0
    partial_result: Optional[Union[List[np.ndarray], np.ndarray]] = None

    def step(self, values) -> None:
        summed_values = np.sum(values, axis=self.axis)

        if self.partial_result is None:
            if axis_is_zero_or_none(self.axis):
                self.partial_result = np.zeros(shape=np.shape(summed_values))
            else:
                self.partial_result = []

        if axis_is_zero_or_none(self.axis):
            self.partial_result += summed_values
        else:
            self.partial_result.append(summed_values)

        if self.axis is None:
            self.n += values.size
        else:
            self.n += values.shape[self.axis]

    def get(self) -> np.ndarray:
        if axis_is_zero_or_none(self.axis):
            return self.partial_result / self.n

        return np.concatenate(self.partial_result) / self.n