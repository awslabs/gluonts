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

from typing import List, Tuple

import numpy as np


class LinearInterpolation:
    """
    Linear interpolation based on datapoints (x_coord, y_coord)

    Parameters
    ----------
    x_coord
        x-coordinates of the data points must be in increasing order.
    y_coord
        y-coordinates of the data points - may be a list of lists.
    tol
        tolerance when performing the division in the linear interpolation.
    """

    def __init__(
        self,
        x_coord: List[float],
        y_coord: List[np.ndarray],
        tol: float = 1e-8,
    ) -> None:
        self.x_coord = x_coord
        assert sorted(self.x_coord) == self.x_coord
        self.y_coord = y_coord
        self.num_points = len(self.x_coord)
        assert (
            self.num_points >= 2
        ), "Need at least two points for linear interpolation."
        self.tol = tol

    def __call__(self, x: float):
        return self.linear_interpolation(x)

    def linear_interpolation(self, x: float) -> np.ndarray:
        """
        If x is out of interpolation range,
        return smallest or largest value.
        Otherwise, find two nearest points [x_1, y_1], [x_2, y_2] and
        return its linear interpolation
        y = (x_2 - x)/(x_2 - x_1) * y_1 + (x - x_1)/(x_2 - x_1) * y_2.

        Parameters
        ----------
        x
            x-coordinate to evaluate the interpolated points.

        Returns
        -------
        np.ndarray
            Interpolated values same shape as self.y_coord

        """
        if self.x_coord[0] >= x:
            return self.y_coord[0]
        elif self.x_coord[-1] <= x:
            return self.y_coord[-1]
        else:
            for i, (x1, x2) in enumerate(zip(self.x_coord, self.x_coord[1:])):
                if x1 < x < x2:
                    denominator = x2 - x1 + self.tol
                    return (x2 - x) / denominator * self.y_coord[i] + (
                        x - x1
                    ) / denominator * self.y_coord[i + 1]


class ExponentialTailApproximation:
    """
    Approximate function on tails based on knots and make a inference on query
    point. Can be used for either interpolation or extrapolation on tails.

    Parameters
    ----------
    x_coord
        x-coordinates of the data points must be in increasing order.
    y_coord
        y-coordinates of the data points - may be a higher numpy array.
    tol
        tolerance when performing the division and computing the log in the
        exponential extrapolation.
    """

    def __init__(
        self,
        x_coord: List[float],
        y_coord: List[np.ndarray],
        tol: float = 1e-8,
    ) -> None:
        self.x_coord = x_coord
        assert sorted(self.x_coord) == self.x_coord
        self.y_coord = y_coord
        self.num_points = len(self.x_coord)
        assert (
            self.num_points >= 2
        ), "Need at least two points for exponential approximation."
        self.tol = tol
        (
            self.beta_inv_left,
            self.beta_inv_right,
        ) = self.init_exponential_tail_weights()

    def init_exponential_tail_weights(self) -> Tuple[float, float]:
        """
        Initialize the weight of exponentially decaying tail functions
        based on two extreme points on the left and right, respectively.

        Returns
        -------
        Tuple
            beta coefficient for left and right tails.
        """
        q_log_diff = np.log(
            (self.x_coord[1] + self.tol) / (self.x_coord[0] + self.tol)
            + self.tol
        )
        y_diff_left = self.y_coord[1] - self.y_coord[0]
        beta_inv_left = y_diff_left / q_log_diff

        z_log_diff = np.log(
            (1 - self.x_coord[-2] + self.tol)
            / (1 - self.x_coord[-1] + self.tol)
            + self.tol
        )  # z = 1/(1-q)
        y_diff_right = self.y_coord[-1] - self.y_coord[-2]
        beta_inv_right = y_diff_right / z_log_diff

        return beta_inv_left, beta_inv_right

    def left(self, x: float) -> np.ndarray:
        """
        Return the inference made on exponentially decaying tail functions
        For left tail, x = exp(beta * (q - alpha))
        For right tail, x = 1 - exp(-beta * (q - alpha))

        E.g. for x = self.x_coord[0] or self.x_coord[1], return value is
        exactly self.y_coord[0] or self.y_coord[1], respectively.

        Parameters
        ----------
        x
            x-coordinate to evaluate the right tail.

        """
        return (
            self.beta_inv_left
            * np.log((x + self.tol) / (self.x_coord[1] + self.tol) + self.tol)
        ) + self.y_coord[1]

    def right(self, x: float) -> np.ndarray:
        """
        Return the inference made on exponentially decaying tail functions
        For left tail, x = exp(beta * (q - alpha))
        For right tail, x = 1 - exp(-beta * (q - alpha))

        E.g. for x = self.x_coord[-1] or self.x_coord[-2] ,
        return value is exactly self.y_coord[-1]
        or self.y_coord[-2] respectively.
        Parameters
        ----------
        x
            x-coordinate to evaluate the right tail.
        """
        return (
            self.beta_inv_right
            * np.log(
                (1 - self.x_coord[-2] + self.tol) / (1 - x + self.tol)
                + self.tol
            )
        ) + self.y_coord[-2]

    def tail_range(self, default_left_tail=0.1, default_right_tail=0.9):
        """
        Return an effective range of left and right tails

        """
        left_tail = max(
            self.x_coord[0],
            min(self.x_coord[1], default_left_tail),
        )
        right_tail = min(
            self.x_coord[-1],
            max(self.x_coord[-2], default_right_tail),
        )
        return left_tail, right_tail
