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

# Third-party imports
import numpy as np


def take_along_axis(a: np.array, indices: np.array, axis: int) -> np.array:
    """
    Monkey patch for np.take_along_axis. Should be removed as soon as we can
    bump up to numpy 1.16

    Parameters
    ----------
    a
        Input array
    indices
        Indices to take over the axis
    axis
        Axis over which to take indices over

    Returns
    -------
    np.array
        Array with values taken along the axis with respect to the input
        indices
    """
    Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis + 1 :]
    J = indices.shape[axis]  # Need not equal M
    out = np.empty(Ni + (J,) + Nk)

    for ii in np.ndindex(Ni):
        for kk in np.ndindex(Nk):
            a_1d = a[ii + np.s_[:,] + kk]
            indices_1d = indices[ii + np.s_[:,] + kk]
            out_1d = out[ii + np.s_[:,] + kk]
            out_1d[:] = a_1d[indices_1d]
    return out
