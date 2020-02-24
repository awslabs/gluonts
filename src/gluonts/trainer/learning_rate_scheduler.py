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

from enum import Enum

# Third-party imports
import mxnet as mx
import numpy as np

from gluonts.core.component import validated
from gluonts import ty


class Objective(str, Enum):
    Min = "min"
    Max = "max"

    def is_better(self, current, best):

        if self == self.Min:
            return current < best
        else:
            return current > best

    def baseline(self):
        if self == self.Min:
            return np.Inf
        else:
            return -np.Inf


class MetricAttentiveScheduler(mx.lr_scheduler.LRScheduler):
    r"""
    This scheduler decreases the learning rate based on the value of some
    validation metric to be optimized (maximized or minimized). The value
    of such metric is provided by calling the `step` method on the scheduler.
    A `patience` parameter must be provided, and the scheduler will reduce
    the learning rate if no improvement in the metric is done before
    `patience` observations of the metric.

    Examples:

        `patience = 0`: learning rate will decrease at every call to
        `step`, regardless of the metric value

        `patience = 1`: learning rate is reduced as soon `step` is called
        with a metric value which does not improve over the best encountered

        `patience = 10`: learning rate is reduced if no improvement in the
        metric is recorded in 10 successive calls to `step`

    Parameters
    ----------
    objective
        String, can either be `"min"` or `"max"`
    patience
        The patience to observe before reducing the learning rate, nonnegative integer.
    base_lr
        Initial learning rate to be used.
    decay_factor
        Factor (between 0 and 1) by which to decrease the learning rate.
    min_lr
        Lower bound for the learning rate, learning rate will never go below `min_lr`
    """

    @validated()
    def __init__(
        self,
        objective: Objective,
        patience: ty.NonNegativeInt,
        base_lr: ty.PositiveFloat = 0.01,
        decay_factor: ty.Interval01 = 0.5,
        min_lr: ty.NonNegativeFloat = 0.0,
    ) -> None:
        super(MetricAttentiveScheduler, self).__init__(base_lr=base_lr)

        assert (
            base_lr > min_lr
        ), f"base_lr should greater than min_lr, {base_lr} <= {min_lr}"

        self.decay_factor = decay_factor
        self.patience = patience
        self.objective = objective
        self.min_lr = min_lr
        self.curr_lr = base_lr

        self.reset()

    def __call__(self, num_update: int) -> float:
        return self.curr_lr

    def reset(self):
        self.curr_lr = self.base_lr
        self.steps_without_progress = 0
        self.current_best = self.objective.baseline()

    def step(self, metric_value: float) -> None:
        """
        Inform scheduler of the metric's new value that is being
        optimized. This method should be invoked at regular intervals (e.g.
        at the end of every epoch, after computing a validation score).

        Parameters
        ----------
        metric_value
            Value of the metric that is being optimized.
        """
        if self.steps_without_progress >= self.patience:
            self.curr_lr = max(
                self.min_lr, self.curr_lr * self.decay_factor
            )
            self.steps_without_progress = 0

        if self.objective.is_better(metric_value, self.current_best):
            self.current_best = metric_value
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
