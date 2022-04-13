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

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
from gluonts.dataset.field_names import FieldName

Item = Dict[str, Any]


class Filter(ABC):
    """
    A filter enables filtering the set of time series contained in a dataset.
    """

    @abstractmethod
    def __call__(self, items: List[Item]) -> List[Item]:
        """
        Filters the given items and returns the ones that should be kept in the
        dataset.

        Args:
            items:  The items to filter.

        Returns:
            The filtered items.
        """


# -------------------------------------------------------------------------------------------------


class ConstantTargetFilter(Filter):
    """
    A filter which removes items having only constant target values.

    This filter should be used whenever metrics such as the MASE are required.
    """

    def __init__(self, prediction_length: int, required_length: int = 0):
        self.prediction_length = prediction_length
        self.required_length = required_length

    def __call__(self, items: List[Item]) -> List[Item]:
        limit = (
            None if self.prediction_length == 0 else -self.prediction_length
        )
        return [
            item
            for item in items
            if len(set(item[FieldName.TARGET][-self.required_length : limit]))
            > 1
        ]


class AbsoluteValueFilter(Filter):
    """
    A filter which removes items having absolute average values of more than
    the provided value.
    """

    def __init__(self, value: float):
        self.value = value

    def __call__(self, items: List[Item]) -> List[Item]:
        return [
            item
            for item in items
            if np.mean(np.abs(item[FieldName.TARGET])) < self.value
        ]


class EndOfSeriesCutFilter(Filter):
    """
    A filter which removes the last `n` time steps from a time series.
    """

    def __init__(self, prediction_length: int):
        self.prediction_length = prediction_length

    def __call__(self, items: List[Item]) -> List[Item]:
        return [
            {
                **item,
                FieldName.TARGET: item[FieldName.TARGET][
                    : -self.prediction_length
                ],
            }
            for item in items
        ]


class MinLengthFilter(Filter):
    """
    A filter which ensures that time series have the specified minimum length.
    """

    def __init__(self, length: int):
        self.length = length

    def __call__(self, items: List[Item]) -> List[Item]:
        return [
            item
            for item in items
            if len(item[FieldName.TARGET]) >= self.length
        ]
