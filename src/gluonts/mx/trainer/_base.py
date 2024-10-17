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
from typing import cast, List, Optional, Union

import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np
from mxnet.metric import ndarray

from gluonts.core.component import validated
from gluonts.dataset.loader import DataLoader
from gluonts.exceptions import GluonTSDataError
from gluonts.gluonts_tqdm import tqdm
from gluonts.mx.context import get_mxnet_context
from gluonts.mx.trainer.callback import Callback, CallbackList
from gluonts.mx.util import HybridContext

from .learning_rate_scheduler import LearningRateReduction
from .model_averaging import SelectNBestMean, save_epoch_info, ModelAveraging

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

    A trainer is mainly defined by two sets of parameters. The first one
    determines the number of examples that the network will be trained on
    (`epochs`, `num_batches_per_epoch`), while the second one specifies how
    the gradient updates are performed (`learning_rate`, `clip_gradient` and
    `weight_decay`).

    Parameters
    ----------
    ctx
    epochs
        Number of epochs that the network will train (default: 100).
    num_batches_per_epoch
        Number of batches at each epoch (default: 50).
    learning_rate
        Initial learning rate (default: :math:`10^{-3}`).
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
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        clip_gradient: float = 10.0,
        weight_decay: float = 1e-8,
        init: Union[str, mx.initializer.Initializer] = "xavier",
        hybridize: bool = True,
        callbacks: Optional[List[Callback]] = None,
        add_default_callbacks: bool = True,
    ) -> None:
        assert (
            0 <= epochs < float("inf")
        ), "The value of `epochs` should be >= 0"
        assert (
            0 < num_batches_per_epoch
        ), "The value of `num_batches_per_epoch` should be > 0"
        assert (
            0 < learning_rate < float("inf")
        ), "The value of `learning_rate` should be > 0"

        assert 0 < clip_gradient, "The value of `clip_gradient` should be > 0"
        assert 0 <= weight_decay, "The value of `weight_decay` should be => 0"

        self.epochs = epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.clip_gradient = clip_gradient
        self.weight_decay = weight_decay
        self.init = init
        self.hybridize = hybridize
        self.ctx = ctx if ctx is not None else get_mxnet_context()
        self.halt = False

        # Make sure callbacks is list -- they are assigned to `self.callbacks`
        # below
        callbacks = callbacks or []

        # TODO the following is done for backwards compatibility. For future
        # versions, add the default callbacks as default arg
        if add_default_callbacks:
            if not any(
                isinstance(callback, ModelAveraging) for callback in callbacks
            ):
                callbacks.append(
                    ModelAveraging(avg_strategy=SelectNBestMean(num_models=1))
                )

            if not any(
                isinstance(callback, LearningRateReduction)
                for callback in callbacks
            ):
                callbacks.append(
                    LearningRateReduction(
                        base_lr=learning_rate,
                        patience=10,
                        objective="min",
                    )
                )

        self.callbacks = CallbackList(callbacks)

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
        is_validation_available = validation_iter is not None

        logger.info("Start model training")

        with tempfile.TemporaryDirectory(
            prefix="gluonts-trainer-temp-"
        ) as gluonts_temp, HybridContext(
            net=net,
            hybridize=self.hybridize,
            static_alloc=True,
            static_shape=True,
        ):

            def base_path() -> str:
                return os.path.join(
                    gluonts_temp,
                    f"{STATE_ARTIFACT_FILE_NAME}_{uuid.uuid4()}",
                )

            best_epoch_info = {
                "params_path": "{}-{}.params".format(base_path(), "init"),
                "epoch_no": -1,
                "score": float("inf"),
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

            first_forward = True

            def loop(  # todo call run epoch
                epoch_no,
                batch_iter,
                num_batches_to_use: Optional[int] = None,
                is_training: bool = True,
            ) -> mx.metric.Loss:
                nonlocal first_forward
                tic = time.time()

                epoch_loss = mx.metric.Loss()

                if is_training:
                    # We should not call this method if we haven't compiled the
                    # network yet. Instead, this callback is called after
                    # network initialization.
                    if not first_forward:
                        self.callbacks.on_train_epoch_start(
                            training_network=net
                        )
                else:
                    self.callbacks.on_validation_epoch_start(
                        training_network=net
                    )

                batch_iter = itertools.islice(batch_iter, num_batches_to_use)
                it = tqdm(batch_iter, total=num_batches_to_use)
                any_batches = False

                for batch_no, batch in enumerate(it, start=1):
                    any_batches = True

                    # `batch` here is expected to be a dictionary whose fields
                    # should correspond 1-to-1 with the network inputs
                    # see below how `batch.values()` is fed into the network
                    if self.halt:
                        break

                    if first_forward:
                        first_forward = False
                        _ = net(*batch.values())

                        self.callbacks.on_network_initializing_end(
                            training_network=net
                        )

                        # Call the batch start callback as the model was not
                        # compiled before
                        self.callbacks.on_train_epoch_start(
                            training_network=net
                        )

                    with mx.autograd.record():
                        # we set the mode explicitly as by default mxnet
                        # assumes predict mode and hence dropout layers are
                        # not used if the mode is not explicitly set to
                        # training
                        mode = (
                            autograd.train_mode
                            if is_training
                            else autograd.predict_mode
                        )
                        with mode():
                            output = net(*batch.values())

                        # network can returns several outputs, the first being
                        # always the loss when having multiple outputs, the
                        # forward returns a list in the case of hybrid and a
                        # tuple otherwise we may wrap network outputs in the
                        # future to avoid this type check
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        batch_size = loss.shape[0]

                    if not np.isfinite(ndarray.sum(loss).asscalar()):
                        logger.warning(
                            "Batch [%d] of Epoch[%d] gave NaN loss and it will"
                            " be ignored",
                            batch_no,
                            epoch_no,
                        )
                        should_continue = True
                    else:
                        if is_training:
                            loss.backward()
                            trainer.step(batch_size)

                            should_continue = (
                                self.callbacks.on_train_batch_end(
                                    training_network=net
                                )
                            )
                        else:
                            should_continue = (
                                self.callbacks.on_validation_batch_end(
                                    training_network=net
                                )
                            )

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
                            f"Number of parameters in {net_name}:"
                            f" {num_model_param}"
                        )
                    if not should_continue:
                        self.halt = True
                        break
                it.close()

                if not any_batches:
                    if is_training:
                        error_data_type = "training"
                    else:
                        error_data_type = "validation"
                    raise GluonTSDataError(
                        "No "
                        + error_data_type
                        + " data batch could be constructed; "
                        "this usually indicates that the "
                        + error_data_type
                        + " dataset "
                        "is empty, or consists of too short series."
                        " If using a random data sampler, this might "
                        "be caused by not taking enough samples."
                    )

                # mark epoch end time and log time cost of current epoch
                if not self.halt:
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

                return epoch_loss

            self.callbacks.on_train_start(max_epochs=self.epochs)

            try:
                for epoch_no in range(self.epochs):
                    if self.halt:
                        logger.info(f"Epoch[{epoch_no}] Interrupting training")
                        break

                    curr_lr = trainer.learning_rate
                    logger.info(
                        f"Epoch[{epoch_no}] Learning rate is {curr_lr}"
                    )

                    epoch_loss = loop(
                        epoch_no,
                        train_iter,
                        num_batches_to_use=self.num_batches_per_epoch,
                    )

                    should_continue = self.callbacks.on_train_epoch_end(
                        epoch_no=epoch_no,
                        epoch_loss=loss_value(epoch_loss),
                        training_network=net,
                        trainer=trainer,
                    )

                    if is_validation_available:
                        epoch_loss = loop(
                            epoch_no, validation_iter, is_training=False
                        )

                        should_continue = (
                            should_continue
                            and self.callbacks.on_validation_epoch_end(
                                epoch_no=epoch_no,
                                epoch_loss=loss_value(epoch_loss),
                                training_network=net,
                                trainer=trainer,
                            )
                        )

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
            except KeyboardInterrupt:
                warnings.warn(
                    "Detected KeyboardInterrupt, attempting graceful "
                    "shutdown..."
                )
                # save model and epoch info
                bp = base_path()
                epoch_info = {
                    "params_path": f"{bp}-0000.params",
                    "epoch_no": epoch_no,
                    "score": loss_value(epoch_loss),
                }

                net.save_parameters(epoch_info["params_path"])
                save_epoch_info(bp, epoch_info)

            self.callbacks.on_train_end(
                training_network=net,
                temporary_dir=gluonts_temp,
                ctx=self.ctx,
            )

            logger.info("End model training")
