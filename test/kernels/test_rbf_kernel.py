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
from mxnet import nd
import pytest

# First-party imports
from gluonts.kernels import RBFKernel


test_cases = [
    # This tests the simple case where the amplitude and length scale parameters are constant
    # and the time series lengths are equal.  The number of features and batch size are both 1.
    (
        nd.array([4, 2, 8]).expand_dims(1),
        nd.array([0, 2, 4]).expand_dims(1),
        nd.array([3]),
        nd.array([2]),
        nd.array([[16, 4, 0], [4, 0, 4], [64, 36, 16]]),
    ),
    # This tests the case where the amplitude and length scale parameters are constant
    # and the time series lengths are equal.  The batch size is fixed to 1.
    # The number of features is now 2.
    (
        nd.array([[0, 1], [2, 3], [3, 0], [4, 2]]),
        nd.array([[3, 2], [0, 2], [1, 1], [2, 0]]),
        nd.array([3]),
        nd.array([2]),
        nd.array([[10, 1, 1, 5], [2, 5, 5, 9], [4, 13, 5, 1], [1, 16, 10, 8]]),
    ),
    # This tests the case where the amplitude and length scale parameters are constant
    # The number of features is 2 and the batch size is 1.
    # The time series lengths are unequal.
    (
        nd.array([[0, 1], [2, 3], [3, 0], [4, 2]]),
        nd.array([[3, 2], [0, 2], [1, 3]]),
        nd.array([3]),
        nd.array([2]),
        nd.array([[10, 1, 5], [2, 5, 1], [4, 13, 13], [1, 16, 10]]),
    ),
    # This tests the general case where the batch size is larger than 1 i.e. there are multiple time series.
    # The amplitude and length scale parameters differ for each batch and are constant per time series.
    # The history lengths of the time series differ, as can occur in the training and test set.
    (
        nd.array([[1, -1, 0], [2, 3, -4]]),
        nd.array([[0, 1, 3], [2, -1, 1], [1, 0, -1], [-1, -2, 3]]),
        nd.array([3, 2.1, 4.2]),
        nd.array([1.3, 2.5, 3.2]),
        nd.array(
            [
                [[14, 2, 2, 14], [57, 41, 19, 83]],
                [[40, 56, 24, 72], [84, 116, 172, 12]],
                [[22, 42, 26, 38], [217, 249, 299, 155]],
            ]
        ),
    ),
]


@pytest.mark.parametrize("x1, x2, amplitude, length_scale, exact", test_cases)
def test_radial_basis_function_kernel(
    x1, x2, amplitude, length_scale, exact
) -> None:
    tol = 1e-5
    batch_size = amplitude.shape[0]
    history_length_1 = x1.shape[0]
    history_length_2 = x2.shape[0]
    num_features = x1.shape[1]
    if batch_size > 1:
        x1 = nd.tile(x1, reps=(batch_size, 1, 1))
        x2 = nd.tile(x2, reps=(batch_size, 1, 1))
        for i in range(1, batch_size):
            x1[i, :, :] = (i + 1) * x1[i, :, :]
            x2[i, :, :] = (i - 3) * x2[i, :, :]
    else:
        x1 = x1.reshape(batch_size, history_length_1, num_features)
        x2 = x2.reshape(batch_size, history_length_2, num_features)
    amplitude = amplitude.reshape(batch_size, 1, 1)
    length_scale = length_scale.reshape(batch_size, 1, 1)
    rbf = RBFKernel(amplitude, length_scale)

    exact = amplitude * nd.exp(-0.5 * exact / length_scale ** 2)

    res = rbf.kernel_matrix(x1, x2)
    assert nd.norm(exact - res) < tol
