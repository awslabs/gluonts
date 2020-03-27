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
import logging
import os
import tempfile
import time
import uuid
from typing import Any, List, NamedTuple, Optional, Union

# Third-party imports
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np

# First-party imports
from gluonts.core.component import get_mxnet_context, validated
from gluonts.core.exception import GluonTSDataError, GluonTSUserError
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.support.util import HybridContext
from gluonts.gluonts_tqdm import tqdm

# Relative imports
from . import learning_rate_scheduler as lrs

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


class BestEpochInfo(NamedTuple):
    params_path: str
    epoch_no: int
    metric_value: float


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
    """

    @validated()
    def __init__(
        self,
        ctx: Optional[mx.Context] = None,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        learning_rate_decay_factor: float = 0.5,
        patience: int = 10,
        minimum_learning_rate: float = 5e-5,
        clip_gradient: float = 10.0,
        weight_decay: float = 1e-8,
        init: Union[str, mx.initializer.Initializer] = "xavier",
        hybridize: bool = True,
    ) -> None:

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
        self.ctx = ctx if ctx is not None else get_mxnet_context()
        self.halt = False

    def set_halt(self, signum: int, stack_frame: Any) -> None:
        logger.info("Received signal: {}".format(signum))
        self.halt = True

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
        input_names: List[str],
        train_iter: TrainDataLoader,
        validation_iter: Optional[ValidationDataLoader] = None,
    ) -> None:  # TODO: we may want to return some training information here
        is_validation_available = validation_iter is not None
        self.halt = False

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
                batch_size = train_iter.batch_size

                best_epoch_info = BestEpochInfo(
                    params_path="%s-%s.params" % (base_path(), "init"),
                    epoch_no=-1,
                    metric_value=np.Inf,
                )

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

                def loop(
                    epoch_no, batch_iter, is_training: bool = True
                ) -> mx.metric.Loss:
                    tic = time.time()

                    epoch_loss = mx.metric.Loss()

                    with tqdm(batch_iter) as it:
                        for batch_no, data_entry in enumerate(it, start=1):
                            if self.halt:
                                break

                            inputs = [data_entry[k] for k in input_names]

                            with mx.autograd.record():
                                output = net(*inputs)

                                # network can returns several outputs, the first being always the loss
                                # when having multiple outputs, the forward returns a list in the case of hybrid and a
                                # tuple otherwise
                                # we may wrap network outputs in the future to avoid this type check
                                if isinstance(output, (list, tuple)):
                                    loss = output[0]
                                else:
                                    loss = output

                            if is_training:
                                loss.backward()
                                trainer.step(batch_size)

                            epoch_loss.update(None, preds=loss)
                            lv = loss_value(epoch_loss)

                            if not np.isfinite(lv):
                                logger.warning(
                                    "Epoch[%d] gave nan loss", epoch_no
                                )
                                return epoch_loss

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
                    return epoch_loss

                for epoch_no in range(self.epochs):
                    if self.halt:
                        logger.info(f"Epoch[{epoch_no}] Interrupting training")
                        break

                    curr_lr = trainer.learning_rate
                    logger.info(
                        f"Epoch[{epoch_no}] Learning rate is {curr_lr}"
                    )

                    epoch_loss = loop(epoch_no, train_iter)
                    if is_validation_available:
                        epoch_loss = loop(
                            epoch_no, validation_iter, is_training=False
                        )

                    should_continue = lr_scheduler.step(loss_value(epoch_loss))
                    if not should_continue:
                        logger.info("Stopping training")
                        break

                    if loss_value(epoch_loss) < best_epoch_info.metric_value:
                        best_epoch_info = BestEpochInfo(
                            params_path="%s-%04d.params"
                            % (base_path(), epoch_no),
                            epoch_no=epoch_no,
                            metric_value=loss_value(epoch_loss),
                        )
                        net.save_parameters(
                            best_epoch_info.params_path
                        )  # TODO: handle possible exception

                    if not trainer.learning_rate == curr_lr:
                        if best_epoch_info.epoch_no == -1:
                            raise GluonTSUserError(
                                "Got NaN in first epoch. Try reducing initial learning rate."
                            )

                        logger.info(
                            f"Loading parameters from best epoch "
                            f"({best_epoch_info.epoch_no})"
                        )
                        net.load_parameters(
                            best_epoch_info.params_path, self.ctx
                        )

                logger.info(
                    f"Loading parameters from best epoch "
                    f"({best_epoch_info.epoch_no})"
                )
                net.load_parameters(best_epoch_info.params_path, self.ctx)

                logger.info(
                    f"Final loss: {best_epoch_info.metric_value} "
                    f"(occurred at epoch {best_epoch_info.epoch_no})"
                )

                # save net parameters
                net.save_parameters(best_epoch_info.params_path)

                logger.info("End model training")
