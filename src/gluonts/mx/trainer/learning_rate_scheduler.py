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
from typing import Dict, Any
from typing_extensions import Literal

import numpy as np
import mxnet as mx
from mxnet import gluon
import mxnet.gluon.nn as nn
from pydantic.dataclasses import dataclass

from gluonts.core.component import validated, logger
from gluonts.exceptions import GluonTSUserError

from .callback import Callback


@dataclass
class Metric:
    best: float

    def update(self, metric: float) -> bool:
        if self.should_update(metric):
            self.best = metric
            return True
        return False

    def should_update(self, metric: float) -> bool:
        raise NotImplementedError


@dataclass
class Min(Metric):
    best: float = np.Inf

    def should_update(self, metric: float) -> bool:
        return metric < self.best


@dataclass
class Max(Metric):
    best: float = -np.Inf

    def should_update(self, metric: float) -> bool:
        return metric > self.best


@dataclass
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

    objective: Literal["min", "max"]
    patience: int
    base_lr: float = 0.01
    decay_factor: float = 0.5
    min_lr: float = 0.0
    metric: Metric = field(init=False)
    current_patience: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        assert (
            self.base_lr > 0
        ), f"base_lr should be positive, got {self.base_lr}"
        assert self.base_lr > self.min_lr, (
            "base_lr should greater than min_lr, "
            f"{self.base_lr} <= {self.min_lr}"
        )

        assert (
            0 < self.decay_factor < 1
        ), f"decay_factor:  0 < {self.decay_factor} < 1"

        assert (
            self.patience >= 0
        ), f"patience should be nonnegative, got {self.patience}"

        super().__init__(base_lr=self.base_lr)
        self.cur_lr = self.base_lr
        self.metric = Min() if self.objective == "min" else Max()

    def __call__(self, num_update: int) -> float:
        return self.cur_lr

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

        if self.metric.update(metric_value):
            self.current_patience = 0
        else:
            self.current_patience += 1

        if self.current_patience >= self.patience or not np.isfinite(
            metric_value
        ):
            self.current_patience = 0

            if self.cur_lr == self.min_lr:
                return False

            self.cur_lr = self.decay_factor * self.cur_lr

            if self.cur_lr < self.min_lr:
                self.cur_lr = self.min_lr

        return True


class LearningRateReduction(Callback):
    r"""
    This Callback decreases the learning rate based on the value of some
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
        objective: str,
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

        self.lr_scheduler = MetricAttentiveScheduler(
            objective=objective,
            patience=patience,
            base_lr=base_lr,
            decay_factor=decay_factor,
            min_lr=min_lr,
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
        if not should_continue:
            print(
                "Early stopping based on learning rate scheduler callback"
                " (min_lr was reached)."
            )
            return False

        pre_step_learning_rate = trainer.learning_rate

        trainer.optimizer.set_learning_rate(
            self.lr_scheduler(trainer.optimizer.num_update)
        )

        if not trainer.learning_rate == pre_step_learning_rate:
            if best_epoch_info["epoch_no"] == -1:
                raise GluonTSUserError(
                    "Got NaN in first epoch. Try reducing initial learning"
                    " rate."
                )

            logger.info(
                "Loading parameters from best epoch "
                f"({best_epoch_info['epoch_no']})"
            )
            training_network.load_parameters(
                best_epoch_info["params_path"], ctx
            )

        return True
