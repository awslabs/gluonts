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


from __future__ import annotations
import time
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

from torch import Tensor
from ..engine.distributed import reduce_value


class Meter(ABC):
    """
    Abstract class for meters used in metric stats.
    Every subclass has cached values for some metric

    Parameters
    ----------
    min_mode: bool or None
        if set to true, the smaller value is considered better and vice versa;
        if set to None, no optimality is defined.
    """

    def __init__(self, min_mode: Optional[bool] = True):
        self.min_mode = min_mode
        if min_mode is True:
            self.best = float("inf")
        elif min_mode is False:
            self.best = float("-inf")
        else:
            self.best = None
        self._initialize()

    @abstractmethod
    def _initialize(self):
        pass

    def restart(self) -> None:
        """
        reset the meter to clear the cached values;

        at the same time, the optimal value so far is compared with current metric and updated
        """
        if self.is_optimal:
            self.best = self.value
        self._initialize()

    def update(self, *args) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self):
        """
        read currently measured metric
        """
        pass

    @property
    def is_optimal(self) -> bool:
        """
        indicates whether the current metric reaches optimality

        Raises
        ------
        TypeError
            if min_mode is None
        """
        if self.min_mode is None:
            raise TypeError("No optimal mode is defined")
        if self.min_mode is True:
            return self.value < self.best
        else:
            return self.value > self.best

    def state_dict(self) -> Dict:
        return {"best": self.best}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.best = state_dict["best"]
        self._initialize()

    def __repr__(self) -> str:
        main_str = (
            f"{self.__class__.__name__}: " f"current_value = {self.value:.4f}"
        )
        if self.best is not None:
            main_str += f", best_value = {self.best:.4f}"
        return main_str


class Timer(Meter):
    """
    Timer to save elapsed time
    """

    def __init__(self):
        super(Timer, self).__init__(None)

    @property
    def value(self) -> float:
        """
        elapsed time in seconds since instantiation or last restart call
        """
        return time.time() - self.start_time

    def _initialize(self) -> None:
        self.start_time = time.time()


class NumericalAverageMeter(Meter):
    """
    Maintains running average for scalar metrics

    Call .restart() to clear added values and to start a new round of averaging
    """

    def _initialize(self) -> None:
        self._cum_values = 0.0
        self._count = 0
        self.cache = 0.0

    def update(self, value: Union[Tensor, float]) -> None:
        """
        add the current value to be further averaged

        Parameters
        ----------
        value : float or scalar Tensor
            the new value
        """
        if isinstance(value, Tensor):
            if value.ndim > 0:
                raise ValueError("A non-scalar tensor is provided.")
            value = value.item()
        self._cum_values += value
        self._count += 1
        self.cache = value

    @property
    def value(self) -> float:
        """
        average of all values added since instantiation or last restart call

        Returns
        -------
        float
            the current average
        """
        if self._count == 0:
            warnings.warn(f"Nothing have been added. Inf is returned.")
            return float("inf")
        else:
            return self._cum_values / self._count


class BatchAverageMeter(NumericalAverageMeter):
    """
    Maintains running average for a stream of batched data, in which the outer dimension is
    assumed to be batches

    Call .restart() to clear added values and to start a new round of averaging
    """

    def __init__(
        self, min_mode: Optional[bool] = True, avg_in_batch: bool = True
    ):
        super(BatchAverageMeter, self).__init__(min_mode)
        self.averaging = avg_in_batch

    def update(self, values: Tensor) -> None:
        """
        add a batch of values to be further averaged

        Parameters
        ----------
        values : Tensor

        Raises
        ------
        ValueError
            if `values` is 0-dimensional
        """
        if values.ndim < 1:
            raise ValueError("A scalar tensor is provided.")
        if self.averaging:
            self._cum_values += values.mean().item() * values.size(0)
        else:
            self._cum_values += values.sum().item()
        self._count += values.size(0)
        self.cache = values.sum().item() / values.size(0)


class MeanDeviationMeter(Meter):
    """
    Maintains a stream of deviation and base values, and compute the ratio of their sums
    e.g. WAPE in demand forecasting.

    MD = \sigma{deviation} / \sigma{base}

    Call .restart() to clear added values and to start a new round of averaging
    """

    def __init__(self):
        super(MeanDeviationMeter, self).__init__(min_mode=True)

    def _initialize(self):
        self._cum_deviation = 0.0
        self._cum_base = 0.0
        self.cache = 0.0

    def update(self, deviation: Tensor, base: Tensor) -> None:
        """
        add new deviation and base values

        Parameters
        ----------
        deviation : Tensor
        base : Tensor

        Raises
        ------
        ValueError
            if deviation and base are not in the same shape
        """
        if deviation.size() != base.size():
            raise ValueError(
                "`deviation` and `base` should matchh in dimensions"
            )
        self._cum_deviation += deviation.sum().item()
        self._cum_base += base.sum().item()
        self.cache = deviation.sum().div(base.sum()).item()

    @property
    def value(self) -> float:
        if self._cum_base == 0.0:
            warnings.warn("Cumulative base is 0. " "Inf is returned.")
            return float("inf")
        else:
            return self._cum_deviation / self._cum_base


class RootMeanSquareDeviationMeter(MeanDeviationMeter):
    """
    Maintains a stream of deviation and base values, and compute the following ratio

    MD = \sqrt{\sigma{deviation^2}} / \sqrt{\sigma{base^2}}

    Call .restart() to clear added values and to start a new round of averaging
    """

    def update(self, deviation: Tensor, base: Tensor) -> None:
        """
        add new deviation and base values

        Parameters
        ----------
        deviation : Tensor
        base : Tensor

        Raises
        ------
        ValueError
            if deviation and base are not in the same shape
        """
        if deviation.size() != base.size():
            raise ValueError(
                "`deviation` and `base` should matchh in dimensions"
            )
        self._cum_deviation += deviation.pow(2).sum().item()
        self._cum_base += base.pow(2).sum().item()
        self.cache = (
            deviation.pow(2)
            .sum()
            .pow(0.5)
            .div(base.pow(2).sum().pow(0.5))
            .item()
        )

    @property
    def value(self) -> float:
        if self._cum_base == 0.0:
            warnings.warn("Cumulative base is 0. " "Inf is returned.")
            return float("inf")
        else:
            return self._cum_deviation**0.5 / self._cum_base**0.5
