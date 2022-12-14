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
from typing import Tuple

import numpy as np


def pad_axis(
    a: np.ndarray, *, axis: int = 0, left: int = 0, right: int = 0, value=0
) -> Tuple[np.ndarray, np.ndarray]:
    """Similar to ``np.pad``, but pads only a single axis using `left` and
    `right` parameters.

    ::

        >>> pad_axis([1, 2, 3, 4], left=2, right=3)
        array([0, 0, 1, 2, 3, 0, 0, 0])

    """
    a = np.array(a)

    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (left, right)
    return np.pad(a, pad_width, constant_values=value)


@dataclass
class AxisView:
    data: np.ndarray
    axis: int

    def __getitem__(self, index):
        slices = [slice(None)] * self.data.ndim
        slices[self.axis] = index

        return self.data[tuple(slices)]


def pad_and_slice(
    data: np.ndarray,
    size: int,
    *,
    axis=0,
    pad_value=0,
    pad_to: str = "left",
    take_from: str = "left",
):
    data = np.array(data)

    pad_length = max(0, size - data.shape[axis])

    if pad_length:
        if pad_to == "left":
            data = pad_axis(data, axis=axis, left=pad_length, value=pad_value)
        else:
            data = pad_axis(data, axis=axis, right=pad_length, value=pad_value)

    if take_from == "left":
        return AxisView(data, axis)[:size]
    else:
        return AxisView(data, axis)[-size:]
