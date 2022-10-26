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


@dataclass
class Aggregation:
    axis: Optional[int] = None

    def step(self, values: np.ndarray) -> None:
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Sum(Aggregation):
    """Map-Reduce way of calculating the sum of a stream of values.

    `partial_result` represents one of two things, depending on the axis:
    Case 1 - axis 0 is aggregated (axis is None or 0):
        In each `step`, sum is being calculated and added to `partial_result`.

    Case 2 - axis 0 is not being aggregated:
        In this case, `partial_result` is a list that in the end gets
        concatenated to a np.ndarray.
    """

    partial_result: Optional[Union[List[np.ndarray], np.ndarray]] = None

    def step(self, values: np.ndarray) -> None:
        summed_values = np.sum(values, axis=self.axis)

        if self.axis is None or self.axis == 0:
            if self.partial_result is None:
                self.partial_result = np.zeros(summed_values.shape)
            self.partial_result += summed_values
        else:
            if self.partial_result is None:
                self.partial_result = []
            self.partial_result.append(summed_values)

    def get(self) -> np.ndarray:
        if self.axis is None or self.axis == 0:
            return np.copy(self.partial_result)

        return np.concatenate(self.partial_result)


@dataclass
class Mean(Aggregation):
    """Map-Reduce way of calculating the mean of a stream of values.

    `partial_result` represents one of two things, depending on the axis:
    Case 1 - axis 0 is aggregated (axis is None or 0):
        First sum values acoording to axis and keep track of number of entries
        summed over (`n`) to divide by in the end.

    Case 2 - axis 0 is not being aggregated:
        In this case, `partial_result` is a list of means that in the end gets
        concatenated to a np.ndarray. As this directly calculates the mean,
        `n` is not used.
    """

    partial_result: Optional[Union[List[np.ndarray], np.ndarray]] = None
    n: int = 0

    def step(self, values: np.ndarray) -> None:
        if self.axis is None or self.axis == 0:
            summed_values = np.sum(values, axis=self.axis)
            if self.partial_result is None:
                self.partial_result = np.zeros(summed_values.shape)

            self.partial_result += summed_values

            invalid_value_count = np.ma.count_masked(values)
            if self.axis is None:
                self.n += values.size - invalid_value_count
            else:
                self.n += values.shape[0] - invalid_value_count
        else:
            if self.partial_result is None:
                self.partial_result = []

            mean_values = np.mean(values, axis=self.axis)

            # TODO: find better solution for when all values are masked
            invalid_value_count = np.ma.count_masked(mean_values)
            if invalid_value_count == mean_values.size:
                mean_values = np.full(mean_values.shape, np.nan)

            self.partial_result.append(mean_values)

    def get(self) -> np.ndarray:
        if self.axis is None or self.axis == 0:
            return self.partial_result / self.n

        return np.concatenate(self.partial_result)
