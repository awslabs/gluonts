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

import logging
import math
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast, List, Optional, Union

import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np
from toolz import take

from gluonts.core.component import validated
from gluonts.dataset.loader import DataLoader
from gluonts.exceptions import GluonTSDataError
from gluonts.gluonts_tqdm import tqdm
from gluonts.itertools import IterableSlice
from gluonts.mx.context import get_mxnet_context
from gluonts.mx.trainer.callback import Callback, CallbackList
from gluonts.mx.util import HybridContext

from .learning_rate_scheduler import LearningRateReduction
from .model_averaging import SelectNBestMean, save_epoch_info, ModelAveraging

logger = logging.getLogger("gluonts").getChild("trainer")


def check_loss_finite(val: float) -> None:
    if not np.isfinite(val):
        raise GluonTSDataError(
            "Encountered invalid loss value! Try reducing the learning rate "
            "or try a different likelihood."
        )


def count_model_params(net: nn.HybridBlock) -> int:
    params = net.collect_params()

    return sum(np.prod(value.shape) for value in params.values())


def loss_value(loss: mx.metric.Loss) -> float:
    # Return first loss-value of `loss`.
    return loss.get()[1]


def get_loss(output):
    # Networks can return several outputs, the first being always the loss when
    # having multiple outputs. `forward()` returns a list in the case of
    # hybrid and a tuple otherwise. We may wrap network outputs in the future
    # to avoid this type check.

    if isinstance(output, (list, tuple)):
        return output[0]

    return output


@dataclass
class EarlyStop(Exception):
    epoch_loss: mx.metric.Loss


@dataclass
class Epoch:
    no: int
    total: int

    @property
    def progress(self):
        digits = int(math.log10(self.total)) + 1
        return ("{no:0%d}/{total}" % digits).format(
            no=self.no + 1, total=self.total
        )


@dataclass
class Loop:
    net: nn.HybridBlock
    callback: Callback

    def callback_on_epoch_start(self, training_network):
        raise NotImplementedError

    def invoke_network(self, batch):
        raise NotImplementedError

    def initialize_network(self, batch):
        pass

    def backward(self, losses):
        pass

    def __call__(  # todo call run epoch
        self,
        epoch: Epoch,
        batches,
    ) -> mx.metric.Loss:

        epoch_loss = mx.metric.Loss()
        tic = time.time()
        it = tqdm(batches)

        # `batch` is expected to be a `dict` whose fields should correspond
        # 1-to-1 with the network inputs, since `batch.values()` is fed into
        # the network.
        for batch_no, batch in enumerate(it, start=1):
            if epoch.no == 0 and batch_no == 1:
                self.initialize_network(batch)

            self.callback_on_epoch_start()

            with autograd.record():
                output = self.invoke_network(batch)

            losses = get_loss(output)
            batch_size = len(losses)

            if not np.isfinite(losses.asnumpy()).all():
                logger.warning(
                    f"Batch [{batch_no}] of Epoch[{epoch.no}] gave NaN loss "
                    "and it will be ignored"
                )
                continue

            self.backward(losses)

            epoch_loss.update(None, losses)

            it.set_postfix(
                {
                    "epoch": epoch.progress,
                    self.loss_name: f"{loss_value(epoch_loss):.2f}",
                }
            )

            if not self.callback_on_batch_end():
                raise EarlyStop(epoch_loss)

        it.close()

        # mark epoch end time and log time cost of current epoch
        toc = time.time()
        logger.info(
            f"Epoch[{epoch.no}] Elapsed time {toc - tic:.3f} seconds",
        )

        logger.info(
            "Epoch[%d] Evaluation metric '%s'=%f",
            epoch.no,
            "epoch_loss",
            loss_value,
        )

        return epoch_loss


@dataclass
class TrainLoop(Loop):
    trainer: mx.gluon.Trainer

    @property
    def loss_name(self):
        return "avg_epoch_loss"

    def backward(self, losses):
        losses.backward()
        self.trainer.step(batch_size=len(losses))

    def invoke_network(self, batch):
        with autograd.train_mode():
            return self.net(*batch.values())

    def initialize_network(self, batch):
        self.net(*batch.values())

        logger.info(
            f"Number of parameters in {self.net.__class__.__name__}:"
            f" {count_model_params(self.net)}"
        )

        self.callback.on_network_initializing_end(training_network=self.net)

    def callback_on_epoch_start(self):
        return self.callback.on_train_epoch_start(training_network=self.net)

    def callback_on_batch_end(self):
        return self.callback.on_train_batch_end(training_network=self.net)


class ValidationLoop(Loop):
    @property
    def loss_name(self):
        return "validation_avg_epoch_loss"

    def invoke_network(self, batch):
        with autograd.predict_mode():
            return self.net(*batch.values())

    def callback_on_epoch_start(self):
        return self.callback.on_validation_epoch_start(
            training_network=self.net
        )

    def callback_on_batch_end(self):
        return self.callback.on_validation_batch_end(training_network=self.net)


class Trainer:
    r"""
    A trainer specifies how a network is going to be trained.

    A trainer is mainly defined by two sets of parameters. The first one
    determines the number of examples that the network will be trained on
    (`epochs`, `num_batches_per_epoch`), while the second one specifies how the
    gradient updates are performed (`learning_rate`,
    `learning_rate_decay_factor`, `patience`, `minimum_learning_rate`,
    `clip_gradient` and `weight_decay`).

    Parameters
    ----------
    ctx
    epochs
        Number of epochs that the network will train (default: 100).
    num_batches_per_epoch
        Number of batches at each epoch (default: 50).
    learning_rate
        Initial learning rate (default: :math:`10^{-3}`).
    learning_rate_decay_factor
        Factor (between 0 and 1) by which to decrease the learning rate
        (default: 0.5).
    patience
        The patience to observe before reducing the learning rate, nonnegative
        integer
        (default: 10).
    minimum_learning_rate
        Lower bound for the learning rate (default: :math:`5\cdot 10^{-5}`).
    clip_gradient
        Maximum value of gradient. The gradient is clipped if it is too large
        (default: 10).
    weight_decay
        The weight decay (or L2 regularization) coefficient. Modifies objective
        by adding a penalty for having large weights (default :math:`10^{-8}`).
    init
        Initializer of the weights of the network (default: "xavier").
    hybridize
        If set to True the network will be hybridized before training
    callbacks
        A list of `gluonts.mx.trainer.callback.Callback` to control the
        training.
    add_default_callbacks
        bool, True by default. If `True`, LearningRateReduction and
        ModelAveragingCallbacks are used in addition to the callbacks specified
        in the callbacks argument. Make sure that you only set this to true if
        you don't specify one of the default callbacks yourself or there will
        be "duplicate callbacks". default callbacks:
        >>> callbacks = [
        ...     ModelAveraging(avg_strategy=SelectNBestMean(num_models=1)),
        ...     LearningRateReduction(
        ...         base_lr=1e-3, # learning_rate
        ...         decay_factor=0.5, # learning_rate_decay_factor
        ...         patience=10, # patience
        ...         min_lr=5e-5, # minimum_learning_rate
        ...         objective="min",
        ...     )
        ... ]
    """

    @validated()
    def __init__(
        self,
        ctx: Optional[mx.Context] = None,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        learning_rate_decay_factor: float = 0.5,
        patience: int = 10,
        minimum_learning_rate: float = 5e-5,
        clip_gradient: float = 10.0,
        weight_decay: float = 1e-8,
        init: Union[str, mx.initializer.Initializer] = "xavier",
        hybridize: bool = True,
        callbacks: Optional[List[Callback]] = None,
        add_default_callbacks: bool = True,
    ) -> None:

        if batch_size is not None:
            warnings.warn(
                "batch_size argument is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            batch_size = 32

        assert isinstance(batch_size, int)

        # TODO param disable_default_callbacks to get backwards compatibility
        # deprecation warnings, in the future, the following callbacks should
        # be controlled by altering callbacks:
        if learning_rate_decay_factor is not None:
            warnings.warn(
                'Trainer argument "learning_rate_decay_factor" is deprecated.'
                " Use callbacks instead.",
                DeprecationWarning,
            )
            assert 0 <= learning_rate_decay_factor < 1, (
                "The value of `learning_rate_decay_factor` should be in the"
                " [0, 1) range"
            )
        if patience is not None:
            warnings.warn(
                'Trainer argument "patience" is deprecated. Use callbacks'
                " instead.",
                DeprecationWarning,
            )
            assert 0 <= patience, "The value of `patience` should be >= 0"
        if minimum_learning_rate:
            warnings.warn(
                'Trainer argument "minimum_learning_rate" is deprecated. Use'
                " callbacks instead.",
                DeprecationWarning,
            )
            assert (
                0 <= minimum_learning_rate
            ), "The value of `minimum_learning_rate` should be >= 0"

        assert (
            0 <= epochs < float("inf")
        ), "The value of `epochs` should be >= 0"
        assert 0 < batch_size, "The value of `batch_size` should be > 0"
        assert (
            0 < num_batches_per_epoch
        ), "The value of `num_batches_per_epoch` should be > 0"
        assert (
            0 < learning_rate < float("inf")
        ), "The value of `learning_rate` should be > 0"

        assert 0 < clip_gradient, "The value of `clip_gradient` should be > 0"
        assert 0 <= weight_decay, "The value of `weight_decay` should be => 0"

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.patience = patience
        self.minimum_learning_rate = minimum_learning_rate
        self.clip_gradient = clip_gradient
        self.weight_decay = weight_decay
        self.init = init
        self.hybridize = hybridize
        self.ctx = ctx if ctx is not None else get_mxnet_context()

        # Make sure callbacks is list -- they are assigned to `self.callbacks`
        # below
        callbacks = callbacks or []

        # TODO the following is done for backwards compatibility. For future
        # versions, add the default callbacks as default arg
        if add_default_callbacks:
            default_callbacks = [
                ModelAveraging(avg_strategy=SelectNBestMean(num_models=1)),
                LearningRateReduction(
                    base_lr=learning_rate,
                    decay_factor=learning_rate_decay_factor,
                    patience=patience,
                    min_lr=minimum_learning_rate,
                    objective="min",
                ),
            ]
            self.callbacks = CallbackList(callbacks + default_callbacks)
        else:
            self.callbacks = CallbackList(callbacks)

    def __call__(
        self,
        net: nn.HybridBlock,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:  # TODO: we may want to return some training information here
        """
        Train a network, given an iterable over training (and optionally
        validation) batches.

        Parameters
        ----------
        net
            Network to be trained. This a Gluon HybridBlock, assumed to produce
            a tensor of loss values as output.
        train_iter
            An iterable over batches to be used for training. Batches are
            assumed to be dictionaries, whose values are MXNet arrays that
            correspond to the network inputs.
        validation_iter
            Similar to `train_iter` but the batches produced here are used to
            compute validation metrics.
        """

        logger.info("Start model training")
        net.initialize(ctx=self.ctx, init=self.init)

        with tempfile.TemporaryDirectory(
            prefix="gluonts-trainer-temp-"
        ) as gluonts_temp, HybridContext(
            net=net,
            hybridize=self.hybridize,
            static_alloc=True,
            static_shape=True,
        ):
            gluonts_temp = Path(gluonts_temp)

            def base_path() -> str:
                return gluonts_temp / f"state_{uuid.uuid4()}"

            best_epoch_info = {
                "params_path": f"{base_path()}-init.params",
                "epoch_no": -1,
                "score": np.Inf,
            }

            optimizer = mx.optimizer.Adam(
                learning_rate=self.learning_rate,
                wd=self.weight_decay,
                clip_gradient=self.clip_gradient,
            )

            trainer = mx.gluon.Trainer(
                net.collect_params(),
                optimizer=optimizer,
                kvstore="device",  # FIXME: initialize properly
            )

            self.callbacks.on_train_start(max_epochs=self.epochs)

            train_loop = TrainLoop(
                trainer=trainer, net=net, callback=self.callbacks
            )
            val_loop = (
                ValidationLoop(net=net, callback=self.callbacks)
                if validation_iter is not None
                else None
            )

            train_batches = IterableSlice(
                train_iter, self.num_batches_per_epoch
            )

            for epoch_no in range(self.epochs):
                epoch = Epoch(epoch_no, self.epochs)

                logger.info(
                    f"Epoch[{epoch.no}] Learning rate is {trainer.learning_rate}"
                )

                epoch_loss = train_loop(epoch, train_batches)

                should_continue = self.callbacks.on_train_epoch_end(
                    epoch_no=epoch.no,
                    epoch_loss=loss_value(epoch_loss),
                    training_network=net,
                    trainer=trainer,
                )

                if val_loop is not None:
                    epoch_loss = val_loop(epoch, validation_iter)

                    should_continue = (
                        should_continue
                        and self.callbacks.on_validation_epoch_end(
                            epoch_no=epoch.no,
                            epoch_loss=loss_value(epoch_loss),
                            training_network=net,
                            trainer=trainer,
                        )
                    )

                # save model and epoch info
                bp = base_path()
                epoch_info = {
                    "params_path": f"{bp}-0000.params",
                    "epoch_no": epoch.no,
                    "score": loss_value(epoch_loss),
                }

                net.save_parameters(
                    epoch_info["params_path"]
                )  # TODO: handle possible exception

                save_epoch_info(bp, epoch_info)

                # update best epoch info
                if loss_value(epoch_loss) < cast(
                    float, best_epoch_info["score"]
                ):
                    best_epoch_info = epoch_info.copy()

                should_continue = (
                    should_continue
                    and self.callbacks.on_epoch_end(
                        epoch_no=epoch_no,
                        epoch_loss=loss_value(epoch_loss),
                        training_network=net,
                        trainer=trainer,
                        best_epoch_info=best_epoch_info,
                        ctx=self.ctx,
                    )
                )

                if not should_continue:
                    logger.info("Stopping training")
                    break

            self.callbacks.on_train_end(
                training_network=net,
                temporary_dir=gluonts_temp,
                ctx=self.ctx,
            )

            logger.info("End model training")
