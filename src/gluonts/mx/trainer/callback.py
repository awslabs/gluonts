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

# Standard library imports
from typing import List, Optional
import logging

# Third-party imports
import numpy as np
import mxnet.gluon.nn as nn
import mxnet as mx
from gluonts.mx.trainer.model_averaging import AveragingStrategy
from gluonts.mx.trainer.model_iteration_averaging import (
    IterationAveragingStrategy,
)
from mxnet import gluon

# First-party imports
from gluonts.evaluation import Evaluator
from gluonts.mx.model.predictor import GluonPredictor
from gluonts.mx.trainer.learning_rate_scheduler import MetricAttentiveScheduler
from gluonts.support.util import copy_parameters

# TODO: documentation


class Callback:
    def __init__(self, **kwargs):
        pass

    def on_network_initializing_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        pass

    def on_train_batch_start(self, training_network: nn.HybridBlock) -> None:
        pass

    def on_validation_batch_start(
        self, training_network: nn.HybridBlock
    ) -> None:
        pass

    def on_train_batch_end(self, training_network: nn.HybridBlock) -> None:
        pass

    def on_train_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        return True

    def on_validation_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        return True

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        return True

    def on_train_end(
        self,
        training_network: nn.HybridBlock,
        temporary_file: str,
        ctx: Optional[mx.context.Context] = None,
    ) -> None:
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: List[Callback], **kwargs):
        self.callbacks = callbacks

    def on_network_initializing_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        for callback in self.callbacks:
            callback.on_network_initializing_end(
                training_network=training_network
            )

    def on_train_batch_start(self, training_network: nn.HybridBlock) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_start(training_network=training_network)

    def on_validation_batch_start(
        self, training_network: nn.HybridBlock
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_batch_start(
                training_network=training_network
            )

    def on_train_batch_end(self, training_network: nn.HybridBlock) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(training_network=training_network)

    def on_train_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        return np.all(
            [
                callback.on_train_epoch_end(
                    epoch_no=epoch_no,
                    epoch_loss=epoch_loss,
                    training_network=training_network,
                    trainer=trainer,
                )
                for callback in self.callbacks
            ]
        )

    def on_validation_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        return np.all(
            [
                callback.on_validation_epoch_end(
                    epoch_no=epoch_no,
                    epoch_loss=epoch_loss,
                    training_network=training_network,
                    trainer=trainer,
                )
                for callback in self.callbacks
            ]
        )

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        return np.all(
            [
                callback.on_epoch_end(
                    epoch_no=epoch_no,
                    epoch_loss=epoch_loss,
                    training_network=training_network,
                    trainer=trainer,
                )
                for callback in self.callbacks
            ]
        )

    def on_train_end(
        self,
        training_network: nn.HybridBlock,
        temporary_file: str,
        ctx: Optional[mx.context.Context] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_end(
                training_network=training_network,
                temporary_file=temporary_file,
                ctx=ctx,
            )


class MetricInferenceEarlyStopping(Callback):
    def __init__(
        self,
        validation_dataset,
        predictor: GluonPredictor,
        evaluator: Evaluator = Evaluator(),
        metric: str = "MSE",
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = True,
        minimize_metric: bool = True,
        restore_best_network: bool = True,
        num_samples: int = 100,
    ):
        assert (
            patience >= 0
        ), "EarlyStopping Callback patience needs to be >= 0"
        assert (
            min_delta >= 0
        ), "EarlyStopping Callback min_delta needs to be >= 0.0"
        assert (
            num_samples >= 1
        ), "EarlyStopping Callback num_samples needs to be >= 1"

        self.validation_dataset = validation_dataset
        self.predictor = predictor
        self.evaluator = evaluator
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_network = restore_best_network
        self.num_samples = num_samples

        if minimize_metric:
            self.best_metric_value = np.inf
            self.is_better = np.less
        else:
            self.best_metric_value = -np.inf
            self.is_better = np.greater

        self.validation_metric_history: List[float] = []
        self.best_network = None
        self.n_stale_epochs = 0

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        should_continue = True
        copy_parameters(training_network, self.predictor.prediction_net)

        from gluonts.evaluation.backtest import make_evaluation_predictions

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.validation_dataset,
            predictor=self.predictor,
            num_samples=self.num_samples,
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)

        agg_metrics, item_metrics = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(self.validation_dataset)
        )
        current_metric_value = agg_metrics[self.metric]
        self.validation_metric_history.append(current_metric_value)

        if self.verbose:
            print(
                f"Validation metric {self.metric}: {current_metric_value}, best: {self.best_metric_value}"
            )

        if self.is_better(current_metric_value, self.best_metric_value):
            self.best_metric_value = current_metric_value

            if self.restore_best_network:
                training_network.save_parameters("best_network.params")

            self.n_stale_epochs = 0
        else:
            self.n_stale_epochs += 1
            if self.n_stale_epochs == self.patience:
                should_continue = False
                print(
                    f"EarlyStopping callback initiated stop of training at epoch {epoch_no}."
                )

                if self.restore_best_network:
                    print(
                        f"Restoring best network from epoch {epoch_no - self.patience}."
                    )
                    training_network.load_parameters("best_network.params")

        return should_continue


class TrainingHistory(Callback):
    def __init__(self):
        self.loss_history = []
        self.validation_loss_history = []

    def on_train_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        self.loss_history.append(epoch_loss)
        return True

    def on_validation_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        self.validation_loss_history.append(epoch_loss)
        return True


class TerminateOnNaN(Callback):
    def on_train_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        is_nan = epoch_loss != epoch_loss
        if is_nan:
            print(
                f"TerminateOnNaN Callback initiated stop of training at epoch {epoch_no}."
            )
            return False
        else:
            return True


class WarmStart(Callback):
    def __init__(self, start_network: nn.HybridBlock):
        self.start_network = start_network

    def on_network_initializing_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        copy_parameters(self.start_network, training_network)


class LearningRateReduction(MetricAttentiveScheduler, Callback):
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

    def __init__(
        self,
        objective: str,
        patience: int,
        base_lr: float = 0.01,
        decay_factor: float = 0.5,
        min_lr: float = 0.0,
    ) -> None:

        super(LearningRateReduction, self).__init__(
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
    ) -> bool:
        should_continue = self.step(metric_value=epoch_loss)
        if not should_continue:
            print(
                "Early stopping based on learning rate scheduler callback (min_lr was reached)."
            )
            return False
        trainer.optimizer.set_learning_rate(self(trainer.optimizer.num_update))

        return True


class ModelIterationAveraging(Callback):
    """

    """

    def __init__(self, avg_strategy: IterationAveragingStrategy):
        self.avg_strategy = avg_strategy

    def on_validation_batch_start(
        self, training_network: nn.HybridBlock
    ) -> None:
        # use averaged model for validation
        self.avg_strategy.load_averaged_model(training_network)

    def on_validation_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        self.avg_strategy.load_cached_model(training_network)
        return True

    def on_train_batch_end(self, training_network: nn.HybridBlock) -> None:

        self.avg_strategy.apply(training_network)

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:

        self.avg_strategy.update_average_trigger(
            metric=epoch_loss, epoch=epoch_no + 1
        )
        # once triggered, update the average immediately
        self.avg_strategy.apply(training_network)
        return True

    def on_train_end(
        self,
        training_network: nn.HybridBlock,
        temporary_file: str,
        ctx: Optional[mx.context.Context] = None,
    ) -> None:

        logging.info("Loading averaged parameters.")
        self.avg_strategy.load_averaged_model(training_network)


class ModelAveraging(Callback):
    """

    """

    def __init__(self, avg_strategy: AveragingStrategy):
        self.avg_strategy = avg_strategy

    def on_train_end(
        self,
        training_network: nn.HybridBlock,
        temporary_file: str,
        ctx: Optional[mx.context.Context] = None,
    ) -> None:
        logging.info("Computing averaged parameters.")
        averaged_params_path = self.avg_strategy.apply(temporary_file)

        logging.info("Loading averaged parameters.")
        training_network.load_parameters(averaged_params_path, ctx)
