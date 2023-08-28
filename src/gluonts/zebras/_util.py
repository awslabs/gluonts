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

import copy
import dataclasses

import numpy as np

from toolz import valmap


@dataclasses.dataclass
class AxisView:
    data: np.ndarray
    axis: int

    def __getitem__(self, index):
        slices = [slice(None)] * self.data.ndim
        slices[self.axis] = index

        return self.data[tuple(slices)]

    def __len__(self):
        return self.data.shape[self.axis]


def pad_axis(
    a: np.ndarray, *, axis: int = 0, left: int = 0, right: int = 0, value=0
) -> np.ndarray:
    """Similar to ``np.pad``, but pads only a single axis using `left` and
    `right` parameters.
    ::
        >>> pad_axis([1, 2, 3, 4], left=2, right=3)
        array([0, 0, 1, 2, 3, 4, 0, 0, 0])
    """
    a = np.array(a)

    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (left, right)
    return np.pad(a, pad_width, constant_values=value)


def _replace(obj, **kwargs):
    """Copy and replace dataclass instance.

    Compared to ``dataclasses.replace`` this first creates a copy where each
    field in the object is copied. Thus, each field of the returned object is
    different from the source object.
    """

    clone = object.__new__(obj.__class__)
    clone.__dict__ = valmap(copy.copy, obj.__dict__)

    return dataclasses.replace(clone, **kwargs)
