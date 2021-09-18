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

import os
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def pad_to_size(
    x: np.array, size: int, axis: int = 0, is_right_pad: bool = True
):
    """Pads `xs` with 0 on the right (default) on the specified axis, which is the first axis by default."""
    pad_length = size - x.shape[axis]
    if pad_length <= 0:
        return x

    pad_width = [(0, 0)] * x.ndim
    right_pad = (0, pad_length)
    pad_width[axis] = right_pad if is_right_pad else right_pad[::-1]
    return np.pad(x, mode="constant", pad_width=pad_width)


class Timer:
    """Context manager for measuring the time of enclosed code fragments."""

    def __enter__(self):
        self.start = time.perf_counter()
        self.interval = None
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class SignalHandler:
    """
    A context manager that attaches a set of signal handlers within its scope.

    Parameters
    ----------
    handlers_map
        A dictionary mapping signal numbers to associated signal handlers to
        be attached within the scope of the enclosing `SignalHandler` instance.
    """

    Callback = Optional[Callable[[int, Any], None]]

    def __init__(self, handlers_map: Dict[int, Callback]) -> None:
        self.handlers_map = handlers_map

    def __enter__(self):
        self.default_handlers = {
            s: signal.signal(s, h) for s, h in self.handlers_map.items()
        }
        return self

    def __exit__(self, *args):
        for s, h in self.default_handlers.items():
            signal.signal(s, h)


def maybe_len(obj) -> Optional[int]:
    try:
        return len(obj)
    except (NotImplementedError, AttributeError):
        return None


def get_download_path() -> Path:
    """

    Returns
    -------
    Path
        default path to download datasets or models of gluon-ts.
        The path is either $MXNET_HOME if the environment variable is defined or
        /home/username/.mxnet/gluon-ts/
    """
    return Path(
        os.environ.get("MXNET_HOME", str(Path.home() / ".mxnet" / "gluon-ts"))
    )


def map_dct_values(fn: Callable, dct: dict) -> dict:
    """Maps `fn` over a dicts values."""
    return {key: fn(value) for key, value in dct.items()}


def erf(x: np.array) -> np.array:
    # Using numerical recipes approximation for erf function
    # accurate to 1E-7

    ones = np.ones_like(x)
    zeros = np.zeros_like(x)

    t = ones / (ones + 0.5 * np.abs(x))

    coefficients = [
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277,
    ]

    inner = zeros
    for c in coefficients[::-1]:
        inner = t * (c + inner)

    res = ones - t * np.exp((inner - 1.26551223 - np.square(x)))
    return np.where(x >= zeros, res, -1.0 * res)


def erfinv(x: np.array) -> np.array:
    zeros = np.zeros_like(x)

    w = -np.log((1.0 - x) * (1.0 + x))
    mask_lesser = w < (zeros + 5.0)

    w = np.where(mask_lesser, w - 2.5, np.sqrt(w) - 3.0)

    coefficients_lesser = [
        2.81022636e-08,
        3.43273939e-07,
        -3.5233877e-06,
        -4.39150654e-06,
        0.00021858087,
        -0.00125372503,
        -0.00417768164,
        0.246640727,
        1.50140941,
    ]

    coefficients_greater_equal = [
        -0.000200214257,
        0.000100950558,
        0.00134934322,
        -0.00367342844,
        0.00573950773,
        -0.0076224613,
        0.00943887047,
        1.00167406,
        2.83297682,
    ]

    p = np.where(
        mask_lesser,
        coefficients_lesser[0] + zeros,
        coefficients_greater_equal[0] + zeros,
    )

    for c_l, c_ge in zip(
        coefficients_lesser[1:], coefficients_greater_equal[1:]
    ):
        c = np.where(mask_lesser, c_l + zeros, c_ge + zeros)
        p = c + p * w

    return p * x


class LinearInterpolation:
    """
    Linear interpolation based on datapoints (x_coord, y_coord)

    Parameters
    ----------
    x_coord
        x-coordinates of the data points must be in increasing order.
    y_coord
        y-coordinates of the data points - may be a list of lists.
    interpolation_type
        Defines type of interpolation to be performed.
    tol
        tolerance when performing the division in the interpolation
    """

    def __init__(
        self,
        x_coord: List[float],
        y_coord: List[np.ndarray],
        interpolation_type: str = "linear",
        tol: float = 1e-8,
    ) -> None:
        self.x_coord = x_coord
        assert sorted(self.x_coord) == self.x_coord
        self.y_coord = y_coord
        self.interpolation_type = interpolation_type
        self.tol = tol

    def __call__(self, x: float):
        if self.interpolation_type == "linear":
            return self.linear_interpolation(x)
        else:
            raise NotImplementedError(
                f"unknown interpolation type {self.interpolation_type}"
            )

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


class TailApproximation:
    """
    Approximate function on tails based on knots and make a inference on query point.
    Can be used for either interpolation or extrapolation on tails

    Parameters
    ----------
    x_coord
        x-coordinates of the data points must be in increasing order.
    y_coord
        y-coordinates of the data points - may be a higher numpy array.
    approximation_type
        str of tail approximation type

    """

    def __init__(
        self,
        x_coord: List[float],
        y_coord: List[np.ndarray],
        approximation_type: str = "exponential",
        tol: float = 1e-8,
    ) -> None:
        self.x_coord = x_coord
        assert sorted(self.x_coord) == self.x_coord
        self.y_coord = y_coord
        self.num_points = len(self.x_coord)
        self.approximation_type = (
            approximation_type if len(self.x_coord) > 1 else None
        )
        self.tol = tol
        if not self.approximation_type:
            pass
        elif self.approximation_type == "exponential":
            self.tail_function_left = self.exponential_tail_left
            self.tail_function_right = self.exponential_tail_right
        else:
            raise NotImplementedError(
                f"unknown approximation type {self.approximation_type}"
            )

    def left(self, x: float) -> np.ndarray:
        """
        Call left tail approximation

        Parameters
        -------
        x
            x-coordinate to evaluate the left tail.

        Returns
        -------
        np.ndarray
            Interpolated values same shape as self.y_coord
        """
        if not self.approximation_type:
            return self.y_coord[0]
        else:
            return self.tail_function_left(x)

    def right(self, x: float) -> np.ndarray:
        """
        Call right tail approximation.

        Parameters
        ----------
        x
            x-coordinate to evaluate the right tail.

        Returns
        -------
        np.ndarray
            Right tail approximated values same shape as self.y_coord.
        """
        if not self.approximation_type:
            return self.y_coord[-1]
        else:
            return self.tail_function_right(x)

    def init_exponential_tail_weights(self) -> Tuple[float, float]:
        """
        Initialize the weight of exponentially decaying tail functions
        based on two extreme points on the left and right, respectively.

        Returns
        -------
        Tuple
            beta coefficient for left and right tails.
        """
        assert (
            self.num_points >= 2
        ), "Need at least two points for exponential approximation"
        q_log_diff = np.log(
            (self.x_coord[1] + self.tol) / (self.x_coord[0] + self.tol)
            + self.tol
        )
        y_diff = self.y_coord[1] - self.y_coord[0]
        beta_inv_left = y_diff / q_log_diff

        z_log_diff = np.log(
            (1 - self.x_coord[-2] + self.tol)
            / (1 - self.x_coord[-1] + self.tol)
            + self.tol
        )  # z = 1/(1-q)
        y_diff = self.y_coord[-1] - self.y_coord[-2]
        beta_inv_right = y_diff / z_log_diff

        return beta_inv_left, beta_inv_right

    def exponential_tail_left(self, x: float) -> np.ndarray:
        """
        Return the inference made on exponentially decaying tail functions
        For left tail, x = exp(beta * (q - alpha))
        For right tail, x = 1 - exp(-beta * (q - alpha))

        E.g. for x = self.x_coord[0] or self.x_coord[1] ,
        return value is exactly self.y_coord[0] or self.y_coord[1], respectively.

        Parameters
        ----------
        x
            x-coordinate to evaluate the right tail.

        """
        beta_inv_left, _ = self.init_exponential_tail_weights()
        return (
            beta_inv_left
            * np.log((x + self.tol) / (self.x_coord[1] + self.tol) + self.tol)
        ) + self.y_coord[1]

    def exponential_tail_right(self, x: float) -> np.ndarray:
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
        _, beta_inv_right = self.init_exponential_tail_weights()
        return (
            beta_inv_right
            * np.log(
                (1 - self.x_coord[-2] + self.tol) / (1 - x + self.tol)
                + self.tol
            )
        ) + self.y_coord[-2]

    def tail_range(self, default_left_tail=0.1, default_right_tail=0.9):
        """
        Return an effective range of left and right tails

        """
        if self.approximation_type == "exponential":
            left_tail = max(
                self.x_coord[0],
                min(self.x_coord[1], default_left_tail),
            )
            right_tail = min(
                self.x_coord[-1],
                max(self.x_coord[-2], default_right_tail),
            )
        else:
            (left_tail, right_tail) = (
                default_left_tail,
                default_right_tail,
            )
        return left_tail, right_tail
