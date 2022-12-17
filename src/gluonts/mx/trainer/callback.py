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

from dataclasses import dataclass, field
from typing import List, Union, Dict, Any, Optional
import logging
import math
import time

# Third-party imports
import mxnet.gluon.nn as nn
import mxnet as mx
from mxnet import gluon
from pydantic import BaseModel, PrivateAttr

# First-party imports
from gluonts.core.component import validated
from gluonts.mx.util import copy_parameters

logger = logging.getLogger(__name__)


class Callback:
    """
    Abstract Callback base class.

    Callbacks control the training of the GluonTS trainer. To write a custom
    Callback, you can subclass Callback and overwrite one or more of the hook
    methods. Hook methods with boolean return value stop the training if False
    is returned.
    """

    def on_train_start(self, max_epochs: int) -> None:
        """
        Hook that is called prior to training. This is the very first hook to
        be called.

        Parameters
        ----------
        max_epochs
            The maximum number of epochs that training is running. The actual
            number of epochs may be fewer if another callback hook stops
            training early.
        """

    def on_network_initializing_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        """
        Hook that is called prior to training, after the training network has
        been initialized. This is the first hook where the network is passed.

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
        Hook that is called prior to each validation epoch. This hook is never
        called if no validation data is available during training.

        Parameters
        ----------
        training_network
            The network that is being trained.
        """

    def on_train_batch_end(self, training_network: nn.HybridBlock) -> bool:
        """
        Hook that is called after each training batch.

        Parameters
        ----------
        training_network
            The network that is being trained.

        Returns
        -------
        bool
            A boolean whether the training should continue. Defaults to `True`.
        """
        return True

    def on_validation_batch_end(
        self, training_network: nn.HybridBlock
    ) -> bool:
        """
        Hook that is called after each validation batch. This hook is never
        called if no validation data is available during training.

        Parameters
        ----------
        training_network
            The network that is being trained.

        Returns
        -------
        bool
            A boolean whether the training should continue. Defaults to `True`.
        """
        return True

    def on_train_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        """
        Hook that is called after each training epoch. This method returns a
        boolean whether training should continue.

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
        Hook that is called after each validation epoch. Similar to
        `on_train_epoch_end`, this method returns a boolean whether training
        should continue. Note that it is always called after
        `on_train_epoch_end` within a single epoch. If `on_train_epoch_end`
        returned `False`, this method will not be called.

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
        `on_validation_epoch_end`, it returns a boolean whether training should
        continue. This hook is always called after `on_train_epoch_end` and
        `on_validation_epoch_end`. It is called regardless of these hooks'
        return values.

        Parameters
        ----------
        epoch_no
            The current epoch (the first epoch has `epoch_no = 0`).
        epoch_loss
            The validation loss that was recorded in the last epoch if
            validation data was provided. The training loss otherwise.
        training_network
            The network that is being trained.
        trainer
            The trainer which is running the training.
        best_epoch_info
            Aggregate information about the best epoch. Contains keys
            `params_path`, `epoch_no` and `score`. The score is the best
            validation loss if validation data is provided or the best training
            loss otherwise.
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
        Hook that is called after training is finished. This is the last hook
        to be called.

        Parameters
        ----------
        training_network
            The network that was trained.
        temporary_dir
            The directory where model parameters are logged throughout
            training.
        ctx
            An MXNet context used.
        """


class CallbackList(Callback):
    """
    Used to chain a list of callbacks to one Callback. Boolean hook methods are
    logically joined with AND, meaning that if at least one callback method
    returns False, the training is stopped.

    Attributes
    ----------
    callbacks
        A list of gluonts.mx.trainer.callback.Callback's.
    """

    @validated()
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def _exec(
        self, function_name: str, *args: Any, **kwargs: Any
    ) -> Union[List[bool], List[None]]:
        return [
            getattr(callback, function_name)(*args, **kwargs)
            for callback in self.callbacks
        ]

    def on_train_start(self, *args: Any, **kwargs: Any) -> None:
        self._exec("on_train_start", *args, **kwargs)

    def on_network_initializing_end(self, *args: Any, **kwargs: Any) -> None:
        self._exec("on_network_initializing_end", *args, **kwargs)

    def on_train_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        self._exec("on_train_epoch_start", *args, **kwargs)

    def on_validation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        self._exec("on_validation_epoch_start", *args, **kwargs)

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> bool:
        return all(self._exec("on_train_batch_end", *args, **kwargs))

    def on_validation_batch_end(self, *args: Any, **kwargs: Any) -> bool:
        return all(self._exec("on_validation_batch_end", *args, **kwargs))

    def on_train_epoch_end(self, *args: Any, **kwargs: Any) -> bool:
        return all(self._exec("on_train_epoch_end", *args, **kwargs))

    def on_validation_epoch_end(self, *args: Any, **kwargs: Any) -> bool:
        return all(self._exec("on_validation_epoch_end", *args, **kwargs))

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> bool:
        return all(self._exec("on_epoch_end", *args, **kwargs))

    def on_train_end(self, *args: Any, **kwargs: Any) -> None:
        self._exec("on_train_end", *args, **kwargs)


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
        if math.isnan(epoch_loss):
            logging.warning(
                "TerminateOnNaN Callback initiated stop of training at epoch"
                f" {epoch_no}."
            )
            return False
        return True


class WarmStart(Callback):
    @validated()
    def __init__(self, predictor):
        self.predictor = predictor

    def on_network_initializing_end(
        self, training_network: nn.HybridBlock
    ) -> None:
        copy_parameters(self.predictor.prediction_net, training_network)


@dataclass
class _Timer:
    duration: float
    _start: Optional[float] = field(init=False, default=None)

    def start(self):
        assert self._start is None

        self._start = time.time()

    def remaining(self) -> float:
        assert self._start is not None

        return max(self._start - time.time(), 0.0)

    def is_running(self) -> bool:
        return self.remaining() > 0


class TrainingTimeLimit(BaseModel, Callback):
    """Limit time spent for training.

    This is useful when ensuring that training for a given model doesn't
    exceed a budget, for example when doing AutoML.

    If `stop_within_epoch` is set to true, training can be stopped after
    each batch, otherwise it stops after the end of the epoch.
    """

    time_limit: float
    stop_within_epoch: bool = False
    _timer: _Timer = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._timer = _Timer(self.time_limit)

    def on_train_start(self, max_epochs: int) -> None:
        self._timer.start()

    def on_train_batch_end(self, training_network: nn.HybridBlock) -> bool:
        if self.stop_within_epoch:
            return self._timer.is_running()

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
        return self._timer.is_running()
