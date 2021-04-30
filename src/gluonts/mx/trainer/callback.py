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
from typing import List, Optional, Union, Dict, Any
import logging

# Third-party imports
import numpy as np
import mxnet.gluon.nn as nn
import mxnet as mx
from gluonts.core.exception import GluonTSUserError
from gluonts.mx.trainer.model_averaging import AveragingStrategy
from gluonts.mx.trainer.model_iteration_averaging import (
    IterationAveragingStrategy,
)
from mxnet import gluon

# First-party imports
from gluonts.core.component import validated, logger
from gluonts.mx.trainer.learning_rate_scheduler import MetricAttentiveScheduler
from gluonts.mx.util import copy_parameters


class Callback:
    """
    Abstract Callback base class.
    Callbacks control the training of the GluonTS trainer.
    To write a custom Callback, you can subclass Callback and overwrite one or more of the hook
    methods. Hook methods with boolean return value stop the training if False is returned.
    """

    @validated()
    def __init__(self, **kwargs):
        pass

    def on_train_start(self, max_epochs: int) -> None:
        """
        Hook that is called prior to training. This is the very first hook to be called.

        Parameters
        ----------
        max_epochs
            The maximum number of epochs that training is running. The actual number of epochs may
            be fewer if another callback hook stops training early.
        """

    def on_network_initializing_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        """
        Hook that is called prior to training, after the training network has been initialized.
        This is the first hook where the network is passed.

        Parameters
        ----------
        training_network
            The network that is being trained.
        """

    def on_train_epoch_start(self, training_network: nn.HybridBlock) -> None:
        """
        Hook that is called prior to each training epoch.

        Parameters
        ----------
        training_network
            The network that is being trained.
        """

    def on_validation_epoch_start(
        self, training_network: nn.HybridBlock
    ) -> None:
        """
        Hook that is called prior to each validation epoch. This hook is never called if no
        validation data is available during training.

        Parameters
        ----------
        training_network
            The network that is being trained.
        """

    def on_train_batch_end(self, training_network: nn.HybridBlock) -> None:
        """
        Hook that is called after each training batch.

        Parameters
        ----------
        training_network
            The network that is being trained.
        """

    def on_validation_batch_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        """
        Hook that is called after each validation batch. This hook is never called if no validation
        data is available during training.

        Parameters
        ----------
        training_network
            The network that is being trained.
        """

    def on_train_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        """
        Hook that is called after each training epoch. This method returns a boolean whether
        training should continue.

        Parameters
        ----------
        epoch_no
            The current epoch (the first epoch has `epoch_no = 0`).
        epoch_loss
            The loss that was recorded in the last epoch.
        training_network
            The network that is being trained.
        trainer
            The trainer which is running the training.

        Returns
        -------
        bool
            A boolean whether the training should continue. Defaults to `True`.
        """
        return True

    def on_validation_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        """
        Hook that is called after each validation epoch. Similar to `on_train_epoch_end`, this
        method returns a boolean whether training should continue. Note that it is always called
        after `on_train_epoch_end` within a single epoch. If `on_train_epoch_end` returned `False`,
        this method will not be called.

        Parameters
        ----------
        epoch_no
            The current epoch (the first epoch has `epoch_no = 0`).
        epoch_loss
            The validation loss that was recorded in the last epoch.
        training_network
            The network that is being trained.
        trainer
            The trainer which is running the training.

        Returns
        -------
        bool
            A boolean whether the training should continue. Defaults to `True`.
        """
        return True

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
        best_epoch_info: Dict[str, Any],
        ctx: mx.Context,
    ) -> bool:
        """
        Hook that is called after every epoch. As `on_train_epoch_end` and
        `on_validation_epoch_end`, it returns a boolean whether training should continue. This
        hook is always called after `on_train_epoch_end` and `on_validation_epoch_end`. It is
        called regardless of these hooks' return values.

        Parameters
        ----------
        epoch_no
            The current epoch (the first epoch has `epoch_no = 0`).
        epoch_loss
            The validation loss that was recorded in the last epoch if validation data was
            provided. The training loss otherwise.
        training_network
            The network that is being trained.
        trainer
            The trainer which is running the training.
        best_epoch_info
            Aggregate information about the best epoch. Contains keys `params_path`, `epoch_no` and
            `score`. The score is the best validation loss if validation data is provided or the
            best training loss otherwise.
        ctx
            The MXNet context used.

        Returns
        -------
        bool
            A boolean whether the training should continue. Defaults to `True`.
        """
        return True

    def on_train_end(
        self,
        training_network: nn.HybridBlock,
        temporary_dir: str,
        ctx: mx.context.Context = None,
    ) -> None:
        """
        Hook that is called after training is finished. This is the last hook to be called.

        Parameters
        ----------
        training_network
            The network that was trained.
        temporary_dir
            The directory where model parameters are logged throughout training.
        ctx
            An MXNet context used.
        """


class CallbackList(Callback):
    """
    Used to chain a list of callbacks to one Callback.
    Boolean hook methods are logically joined with AND, meaning that if at least one callback method returns False, the training is stopped.

    Parameters
    ----------
    callbacks
        A list of gluonts.mx.trainer.callback.Callback's.
    """

    @validated()
    def __init__(self, callbacks: Union[List[Callback], Callback], **kwargs):

        self.callbacks = (
            callbacks if isinstance(callbacks, list) else [callbacks]
        )

    def include(
        self, new_callbacks: Union["CallbackList", List[Callback]]
    ) -> None:
        """
        Include callbacks of a CallbackList or a list of callbacks in self.callbacks.
        If two Callbacks are the same type, self.callbacks are prioritized and the new clalback will not be added.

        Parameters
        ----------
        callbacks
            A gluonts.mx.trainer.callback.CallbackList.
        """

        if not isinstance(new_callbacks, list):
            new_callbacks = new_callbacks.callbacks

        callback_types = set([type(callback) for callback in self.callbacks])
        # make sure not to have no duplicates
        for callback in new_callbacks:
            if type(callback) in callback_types:
                continue
            else:
                self.callbacks.append(callback)

    def on_train_start(self, max_epochs: int) -> None:
        for callback in self.callbacks:
            callback.on_train_start(max_epochs=max_epochs)

    def on_network_initializing_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        for callback in self.callbacks:
            callback.on_network_initializing_end(
                training_network=training_network
            )

    def on_train_epoch_start(self, training_network: nn.HybridBlock) -> None:
        for callback in self.callbacks:
            callback.on_train_epoch_start(training_network=training_network)

    def on_validation_epoch_start(
        self, training_network: nn.HybridBlock
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_epoch_start(
                training_network=training_network
            )

    def on_train_batch_end(self, training_network: nn.HybridBlock) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(training_network=training_network)

    def on_validation_batch_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_batch_end(training_network=training_network)

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
        best_epoch_info: Dict[str, Any],
        ctx: mx.Context,
    ) -> bool:
        return np.all(
            [
                callback.on_epoch_end(
                    epoch_no=epoch_no,
                    epoch_loss=epoch_loss,
                    training_network=training_network,
                    trainer=trainer,
                    best_epoch_info=best_epoch_info,
                    ctx=ctx,
                )
                for callback in self.callbacks
            ]
        )

    def on_train_end(
        self,
        training_network: nn.HybridBlock,
        temporary_dir: str,
        ctx: mx.context.Context = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_end(
                training_network=training_network,
                temporary_dir=temporary_dir,
                ctx=ctx,
            )


class TrainingHistory(Callback):
    @validated()
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
    @validated()
    def __init__(self, predictor):
        self.predictor = predictor

    def on_network_initializing_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        copy_parameters(self.predictor.prediction_net, training_network)


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
        assert 0 <= patience, "The value of `patience` should be >= 0"
        assert (
            0 <= min_lr <= base_lr
        ), "The value of `min_lr` should be >= 0 and <= base_lr"

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
        best_epoch_info: Dict[str, Any],
        ctx: mx.Context,
    ) -> bool:
        should_continue = self.step(metric_value=epoch_loss)
        if not should_continue:
            print(
                "Early stopping based on learning rate scheduler callback (min_lr was reached)."
            )
            return False
        pre_step_learning_rate = trainer.learning_rate
        trainer.optimizer.set_learning_rate(self(trainer.optimizer.num_update))

        if not trainer.learning_rate == pre_step_learning_rate:
            if best_epoch_info["epoch_no"] == -1:
                raise GluonTSUserError(
                    "Got NaN in first epoch. Try reducing initial learning rate."
                )

            logger.info(
                f"Loading parameters from best epoch "
                f"({best_epoch_info['epoch_no']})"
            )
            training_network.load_parameters(
                best_epoch_info["params_path"], ctx
            )

        return True


class ModelIterationAveraging(Callback):
    """
    Callback to implement iteration based model averaging strategies.

    Parameters
    ----------
    avg_strategy
        IterationAveragingStrategy, one of NTA or Alpha_Suffix from gluonts.mx.trainer.model_iteration_averaging
    """

    @validated()
    def __init__(self, avg_strategy: IterationAveragingStrategy):
        self.avg_strategy = avg_strategy

    def on_validation_epoch_start(
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
        best_epoch_info: Dict[str, Any],
        ctx: mx.Context,
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
        temporary_dir: str,
        ctx: mx.context.Context = None,
    ) -> None:

        logging.info("Loading averaged parameters.")
        self.avg_strategy.load_averaged_model(training_network)


class ModelAveraging(Callback):
    """
    Callback to implement model averaging strategies.
    Selects the checkpoints with the best loss values and computes the model average or weighted model average depending on the chosen avg_strategy.


    Parameters
    ----------
    avg_strategy
        AveragingStrategy, one of SelectNBestSoftmax or SelectNBestMean from gluonts.mx.trainer.model_averaging
    """

    @validated()
    def __init__(self, avg_strategy: AveragingStrategy):
        self.avg_strategy = avg_strategy

    def on_train_end(
        self,
        training_network: nn.HybridBlock,
        temporary_dir: str,
        ctx: mx.context.Context = None,
    ) -> None:
        logging.info("Computing averaged parameters.")
        averaged_params_path = self.avg_strategy.apply(temporary_dir)

        logging.info("Loading averaged parameters.")
        training_network.load_parameters(averaged_params_path, ctx)
