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

import itertools
import logging
import os
import tempfile
import time
import uuid
import warnings
from typing import Any, Callable, List, Optional, Union

import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np
from mxnet.metric import ndarray

from gluonts.core.component import validated
from gluonts.core.exception import GluonTSDataError, GluonTSUserError
from gluonts.dataset.loader import DataLoader
from gluonts.gluonts_tqdm import tqdm
from gluonts.mx.context import get_mxnet_context
from gluonts.mx.util import HybridContext

from . import learning_rate_scheduler as lrs
from .model_averaging import (
    AveragingStrategy,
    SelectNBestMean,
    save_epoch_info,
)
from .model_iteration_averaging import IterationAveragingStrategy

logger = logging.getLogger("gluonts").getChild("trainer")


MODEL_ARTIFACT_FILE_NAME = "model"
STATE_ARTIFACT_FILE_NAME = "state"

# make the IDE happy: mx.py does not explicitly import autograd
mx.autograd = autograd


def check_loss_finite(val: float) -> None:
    if not np.isfinite(val):
        raise GluonTSDataError(
            "Encountered invalid loss value! Try reducing the learning rate "
            "or try a different likelihood."
        )


def loss_value(loss: mx.metric.Loss) -> float:
    return loss.get_name_value()[0][1]


class Trainer:
    r"""
    A trainer specifies how a network is going to be trained.

    A trainer is mainly defined by two sets of parameters. The first one determines the number of examples
    that the network will be trained on (`epochs`, `num_batches_per_epoch` and `batch_size`), while the
    second one specifies how the gradient updates are performed (`learning_rate`, `learning_rate_decay_factor`,
    `patience`, `minimum_learning_rate`, `clip_gradient` and `weight_decay`).

    Parameters
    ----------
    ctx
    epochs
        Number of epochs that the network will train (default: 100).
    batch_size
        Number of examples in each batch (default: 32).
    num_batches_per_epoch
        Number of batches at each epoch (default: 50).
    learning_rate
        Initial learning rate (default: :math:`10^{-3}`).
    learning_rate_decay_factor
        Factor (between 0 and 1) by which to decrease the learning rate (default: 0.5).
    patience
        The patience to observe before reducing the learning rate, nonnegative integer (default: 10).
    minimum_learning_rate
        Lower bound for the learning rate (default: :math:`5\cdot 10^{-5}`).
    clip_gradient
        Maximum value of gradient. The gradient is clipped if it is too large (default: 10).
    weight_decay
        The weight decay (or L2 regularization) coefficient. Modifies objective by adding a penalty for having
        large weights (default :math:`10^{-8}`).
    init
        Initializer of the weights of the network (default: "xavier").
    hybridize
        If set to true the network will be hybridized before training
    post_initialize_cb
        An optional callback function. If provided the function will be called with the
        initialized network `post_initialize_cb(net)` before the training starts.
        This callback can be used to e.g. overwrite parameters for warm starting, to freeze some
        of the network parameters etc.
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
        avg_strategy: Union[
            AveragingStrategy, IterationAveragingStrategy
        ] = SelectNBestMean(num_models=1),
        post_initialize_cb: Optional[Callable[[mx.gluon.Block], None]] = None,
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
        assert (
            0 <= learning_rate_decay_factor < 1
        ), "The value of `learning_rate_decay_factor` should be in the [0, 1) range"
        assert 0 <= patience, "The value of `patience` should be >= 0"
        assert (
            0 <= minimum_learning_rate
        ), "The value of `minimum_learning_rate` should be >= 0"
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
        self.avg_strategy = avg_strategy
        self.ctx = ctx if ctx is not None else get_mxnet_context()
        self.post_initialize_cb = post_initialize_cb

    def count_model_params(self, net: nn.HybridBlock) -> int:
        params = net.collect_params()
        num_params = 0
        for p in params:
            v = params[p]
            num_params += np.prod(v.shape)
        return num_params

    def __call__(
        self,
        net: nn.HybridBlock,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:  # TODO: we may want to return some training information here
        """
        Train a network, given an iterable over training (and optionally validation) batches.

        Parameters
        ----------
        net
            Network to be trained. This a Gluon HybridBlock, assumed to produce a tensor
            of loss values as output.
        train_iter
            An iterable over batches to be used for training. Batches are assumed to be
            dictionaries, whose values are MXNet arrays that correspond to the network
            inputs.
        validation_iter
            Similar to `train_iter` but the batches produced here are used to compute
            validation metrics.
        """
        is_validation_available = validation_iter is not None

        with tempfile.TemporaryDirectory(
            prefix="gluonts-trainer-temp-"
        ) as gluonts_temp:

            def base_path() -> str:
                return os.path.join(
                    gluonts_temp,
                    "{}_{}".format(STATE_ARTIFACT_FILE_NAME, uuid.uuid4()),
                )

            logger.info("Start model training")

            net.initialize(ctx=self.ctx, init=self.init)

            with HybridContext(
                net=net,
                hybridize=self.hybridize,
                static_alloc=True,
                static_shape=True,
            ):
                best_epoch_info = {
                    "params_path": "%s-%s.params" % (base_path(), "init"),
                    "epoch_no": -1,
                    "score": np.Inf,
                }

                lr_scheduler = lrs.MetricAttentiveScheduler(
                    objective="min",
                    patience=self.patience,
                    decay_factor=self.learning_rate_decay_factor,
                    min_lr=self.minimum_learning_rate,
                )

                optimizer = mx.optimizer.Adam(
                    learning_rate=self.learning_rate,
                    lr_scheduler=lr_scheduler,
                    wd=self.weight_decay,
                    clip_gradient=self.clip_gradient,
                )

                trainer = mx.gluon.Trainer(
                    net.collect_params(),
                    optimizer=optimizer,
                    kvstore="device",  # FIXME: initialize properly
                )

                first_forward = True

                def loop(
                    epoch_no,
                    batch_iter,
                    num_batches_to_use: Optional[int] = None,
                    is_training: bool = True,
                ) -> mx.metric.Loss:
                    nonlocal first_forward
                    tic = time.time()

                    epoch_loss = mx.metric.Loss()

                    # use averaged model for validation
                    if not is_training and isinstance(
                        self.avg_strategy, IterationAveragingStrategy
                    ):
                        self.avg_strategy.load_averaged_model(net)

                    batch_iter = itertools.islice(
                        batch_iter, num_batches_to_use
                    )

                    with tqdm(batch_iter, total=num_batches_to_use) as it:
                        for batch_no, batch in enumerate(it, start=1):
                            # `batch` here is expected to be a dictionary whose fields
                            # should correspond 1-to-1 with the network inputs
                            # see below how `batch.values()` is fed into the network

                            if first_forward:
                                first_forward = False
                                _ = net(*batch.values())
                                if self.post_initialize_cb:
                                    self.post_initialize_cb(net)

                            with mx.autograd.record():
                                # we set the mode explicitly as by default mxnet assumes predict mode and hence
                                # dropout layers are not used if the mode is not explicitly set to training
                                mode = (
                                    autograd.train_mode
                                    if is_training
                                    else autograd.predict_mode
                                )
                                with mode():
                                    output = net(*batch.values())

                                # network can returns several outputs, the first being always the loss
                                # when having multiple outputs, the forward returns a list in the case of hybrid and a
                                # tuple otherwise
                                # we may wrap network outputs in the future to avoid this type check
                                if isinstance(output, (list, tuple)):
                                    loss = output[0]
                                else:
                                    loss = output

                                batch_size = loss.shape[0]

                            if not np.isfinite(ndarray.sum(loss).asscalar()):
                                logger.warning(
                                    "Batch [%d] of Epoch[%d] gave NaN loss and it will be ignored",
                                    batch_no,
                                    epoch_no,
                                )
                            else:
                                if is_training:
                                    loss.backward()
                                    trainer.step(batch_size)

                                    # iteration averaging in training
                                    if isinstance(
                                        self.avg_strategy,
                                        IterationAveragingStrategy,
                                    ):
                                        self.avg_strategy.apply(net)

                                epoch_loss.update(None, preds=loss)

                            lv = loss_value(epoch_loss)
                            it.set_postfix(
                                ordered_dict={
                                    "epoch": f"{epoch_no + 1}/{self.epochs}",
                                    ("" if is_training else "validation_")
                                    + "avg_epoch_loss": lv,
                                },
                                refresh=False,
                            )
                            # print out parameters of the network at the first pass
                            if batch_no == 1 and epoch_no == 0:
                                net_name = type(net).__name__
                                num_model_param = self.count_model_params(net)
                                logger.info(
                                    f"Number of parameters in {net_name}: {num_model_param}"
                                )
                    # mark epoch end time and log time cost of current epoch
                    toc = time.time()
                    logger.info(
                        "Epoch[%d] Elapsed time %.3f seconds",
                        epoch_no,
                        (toc - tic),
                    )

                    logger.info(
                        "Epoch[%d] Evaluation metric '%s'=%f",
                        epoch_no,
                        ("" if is_training else "validation_") + "epoch_loss",
                        lv,
                    )

                    if not is_training and isinstance(
                        self.avg_strategy, IterationAveragingStrategy
                    ):
                        # bring back the cached model
                        self.avg_strategy.load_cached_model(net)

                    return epoch_loss

                for epoch_no in range(self.epochs):

                    curr_lr = trainer.learning_rate
                    logger.info(
                        f"Epoch[{epoch_no}] Learning rate is {curr_lr}"
                    )

                    epoch_loss = loop(
                        epoch_no,
                        train_iter,
                        num_batches_to_use=self.num_batches_per_epoch,
                    )
                    if is_validation_available:
                        epoch_loss = loop(
                            epoch_no, validation_iter, is_training=False
                        )

                    # update average trigger
                    if isinstance(
                        self.avg_strategy, IterationAveragingStrategy
                    ):
                        self.avg_strategy.update_average_trigger(
                            metric=loss_value(epoch_loss), epoch=epoch_no + 1
                        )
                        # once triggered, update the average immediately
                        self.avg_strategy.apply(net)

                    should_continue = lr_scheduler.step(loss_value(epoch_loss))
                    if isinstance(
                        self.avg_strategy, IterationAveragingStrategy
                    ):
                        logging.info(
                            "Overriding early stopping for iteration-based averaging strategies."
                        )
                        should_continue = True
                    if not should_continue:
                        logger.info("Stopping training")
                        break

                    # save model and epoch info
                    bp = base_path()
                    epoch_info = {
                        "params_path": f"{bp}-0000.params",
                        "epoch_no": epoch_no,
                        "score": loss_value(epoch_loss),
                    }

                    net.save_parameters(
                        epoch_info["params_path"]
                    )  # TODO: handle possible exception

                    save_epoch_info(bp, epoch_info)

                    # update best epoch info - needed for the learning rate scheduler
                    if loss_value(epoch_loss) < best_epoch_info["score"]:
                        best_epoch_info = epoch_info.copy()

                    if not trainer.learning_rate == curr_lr:
                        if best_epoch_info["epoch_no"] == -1:
                            raise GluonTSUserError(
                                "Got NaN in first epoch. Try reducing initial learning rate."
                            )

                        logger.info(
                            f"Loading parameters from best epoch "
                            f"({best_epoch_info['epoch_no']})"
                        )
                        net.load_parameters(
                            best_epoch_info["params_path"], self.ctx
                        )

                if isinstance(self.avg_strategy, AveragingStrategy):
                    logging.info("Computing averaged parameters.")
                    averaged_params_path = self.avg_strategy.apply(
                        gluonts_temp
                    )

                    logging.info("Loading averaged parameters.")
                    net.load_parameters(averaged_params_path, self.ctx)

                if isinstance(self.avg_strategy, IterationAveragingStrategy):
                    logging.info("Loading averaged parameters.")
                    self.avg_strategy.load_averaged_model(net)

                logger.info("End model training")
