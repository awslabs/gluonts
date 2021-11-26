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

from gluonts.model.util import (
    LinearInterpolation,
    ExponentialTailApproximation,
)


def test_linear_interpolation() -> None:
    tol = 1e-7
    x_coord = [0.1, 0.5, 0.9]
    y_coord = [
        np.array([0.1, 0.5, 1]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.9]),
    ]
    linear_interpolation = LinearInterpolation(x_coord, y_coord)
    x = 0.75
    exact = y_coord[1] + (x - x_coord[1]) * (y_coord[2] - y_coord[1]) / (
        x_coord[2] - x_coord[1]
    )
    assert np.all(np.abs(exact - linear_interpolation(x)) <= tol)


def test_exponential_left_tail_approximation() -> None:
    tol = 1e-5
    x_coord = [0.1, 0.5, 0.9]
    y_coord = [
        np.array([0.1, 0.5, 1]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.9]),
    ]
    x = 0.01
    beta_inv = np.array([0.55920144, 0.9320024, 1.24266987])
    exact = beta_inv * np.log(x / x_coord[1]) + y_coord[1]
    exp_tail_approximation = ExponentialTailApproximation(x_coord, y_coord)
    assert np.all(np.abs(exact - exp_tail_approximation.left(x)) <= tol)


def test_exponential_right_tail_approximation() -> None:
    tol = 1e-5
    x_coord = [0.1, 0.5, 0.9]
    y_coord = [
        np.array([0.1, 0.5, 1]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.9]),
    ]
    x = 0.99
    beta_inv = np.array([-0.4660012, -0.9320024, -1.30480336])
    exact = beta_inv * np.log((1 - x_coord[1]) / (1 - x)) + y_coord[1]
    exp_tail_approximation = ExponentialTailApproximation(x_coord, y_coord)
    assert np.all(np.abs(exact - exp_tail_approximation.right(x)) <= tol)
