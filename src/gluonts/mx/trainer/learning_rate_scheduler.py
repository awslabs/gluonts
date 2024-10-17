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

from dataclasses import field
from typing import Dict, Any, Optional

from typing_extensions import Literal

import numpy as np
import mxnet as mx
from mxnet import gluon
import mxnet.gluon.nn as nn

from gluonts.core.component import validated
from gluonts.pydantic import dataclass

from .callback import Callback


@dataclass
class Objective:
    best: float

    @staticmethod
    def from_str(s: Literal["min", "max"]) -> "Objective":
        if s == "min":
            return Min()
        else:
            return Max()

    def update(self, metric: float) -> bool:
        if self.should_update(metric):
            self.best = metric
            return True
        return False

    def should_update(self, metric: float) -> bool:
        raise NotImplementedError


@dataclass
class Min(Objective):
    best: float = float("inf")

    def should_update(self, metric: float) -> bool:
        return metric < self.best


@dataclass
class Max(Objective):
    best: float = -float("inf")

    def should_update(self, metric: float) -> bool:
        return metric > self.best


@dataclass
class Patience:
    """
    Simple patience tracker.

    Given an `Objective`, it will check whether the metric has improved and
    update its patience count. A better value sets the patience back to zero.

    In addition, one needs to call ``reset()`` explicitly after the patience
    was exceeded, otherwise `RuntimError` is raised when trying to invoke
    `step`.

    ``Patience`` keeps track of the number of invocations to ``reset``, via
    ``num_resets``.
    """

    patience: int = field(metadata={"ge": 0})
    objective: Objective
    current_patience: int = field(default=0, init=False)
    num_resets: int = field(default=0, init=False)
    exceeded: bool = field(default=False, init=False)

    def reset(self) -> None:
        self.current_patience = 0
        self.exceeded = False
        self.num_resets += 1

    def step(self, metric_value: float) -> bool:
        if self.exceeded:
            raise RuntimeError("Patience already exceeded.")

        has_improved = self.objective.update(metric_value)

        if has_improved:
            self.current_patience = 0
        else:
            self.current_patience += 1

        # this can also trigger in case of improvement when `self.patience = 0`
        self.exceeded = self.current_patience >= self.patience
        return self.exceeded


@dataclass
class MetricAttentiveScheduler:
    """
    This scheduler decreases the learning rate based on the value of some
    validation metric to be optimized (maximized or minimized). The value of
    such metric is provided by calling the `step` method on the scheduler. A
    `patience` parameter must be provided, and the scheduler will reduce the
    learning rate if no improvement in the metric is done before `patience`
    observations of the metric.

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
        The patience to observe before reducing the learning rate, nonnegative
        integer.
    base_lr
        Initial learning rate to be used.
    decay_factor
        Factor (between 0 and 1) by which to decrease the learning rate.
    min_learning_rate
        Lower bound for the learning rate, learning rate will never go below
        `min_learning_rate`.
    """

    patience: Patience
    learning_rate: float = field(default=0.01, metadata={"gt": 0})
    decay_factor: float = field(default=0.5, metadata={"gt": 0, "lt": 1})
    min_learning_rate: float = 0.0
    max_num_decays: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.learning_rate > self.min_learning_rate

    def step(self, metric_value: float) -> bool:
        """
        Inform the scheduler of the new value of the metric that is being
        optimized. This method should be invoked at regular intervals (e.g. at
        the end of every epoch, after computing a validation score).

        Parameters
        ----------
        metric_value
            Value of the metric that is being optimized.

        Returns
        -------
        bool value indicating, whether to continue training
        """

        self.patience.step(metric_value)

        should_continue = True

        if self.patience.exceeded or not np.isfinite(metric_value):
            if (
                self.learning_rate == self.min_learning_rate
                or self.max_num_decays is not None
                and self.max_num_decays <= self.patience.num_resets
            ):
                should_continue = False

            # Even though we ask not to continue, we still reset the patience
            # because we might still end up continuing training. (Can Happen
            # in testing).
            self.patience.reset()

            self.learning_rate *= self.decay_factor
            # ensure that we don't go below the minimum learning rate
            if self.learning_rate < self.min_learning_rate:
                self.learning_rate = self.min_learning_rate

        return should_continue


class LearningRateReduction(Callback):
    """
    This Callback decreases the learning rate based on the value of some
    validation metric to be optimized (maximized or minimized). The value of
    such metric is provided by calling the `step` method on the scheduler. A
    `patience` parameter must be provided, and the scheduler will reduce the
    learning rate if no improvement in the metric is done before `patience`
    observations of the metric.

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
        String, can either be `"min"` or `"max"`.
    patience
        The patience to observe before reducing the learning rate, nonnegative
        integer.
    base_lr
        Initial learning rate to be used.
    decay_factor
        Factor (between 0 and 1) by which to decrease the learning rate.
    min_lr
        Lower bound for the learning rate, learning rate will never go below
        `min_lr`.
    """

    @validated()
    def __init__(
        self,
        objective: Literal["min", "max"],
        patience: int,
        base_lr: float = 0.01,
        decay_factor: float = 0.5,
        min_lr: float = 0.0,
    ) -> None:
        assert (
            0 < decay_factor < 1
        ), "The value of `decay_factor` should be in the (0, 1) range"
        assert patience >= 0, "The value of `patience` should be >= 0"
        assert (
            0 <= min_lr <= base_lr
        ), "The value of `min_lr` should be >= 0 and <= base_lr"

        self.lr_scheduler = MetricAttentiveScheduler(  # type: ignore[call-arg]
            patience=Patience(  # type: ignore[call-arg]
                patience=patience, objective=Objective.from_str(objective)
            ),
            learning_rate=base_lr,
            decay_factor=decay_factor,
            min_learning_rate=min_lr,
        )

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
        best_epoch_info: Dict[str, Any],
        ctx: mx.Context,
    ) -> bool:
        should_continue = self.lr_scheduler.step(metric_value=epoch_loss)
        trainer.optimizer.set_learning_rate(self.lr_scheduler.learning_rate)
        return should_continue
