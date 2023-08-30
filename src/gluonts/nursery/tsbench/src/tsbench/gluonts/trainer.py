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
import time
from typing import List, Optional
import mxnet as mx
import numpy as np
from gluonts.core.component import validated
from gluonts.dataset.loader import DataLoader
from gluonts.gluonts_tqdm import tqdm
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import HybridContext
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.metric import ndarray
from .callbacks import Callback, CallbackList

logger = logging.getLogger("gluonts_meta").getChild("trainer")

# make the IDE happy: mx.py does not explicitly import autograd
mx.autograd = autograd  # type: ignore


class TimedTrainer(Trainer):
    """
    A custom trainer whose training duration is based on wall clock time
    instead of epochs.
    """

    @validated()
    def __init__(
        self,
        training_time: float,
        validation_milestones: Optional[List[float]] = None,
        learning_rate: float = 1e-3,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate)

        validation_milestones = validation_milestones or []
        assert all(
            x < y
            for x, y in zip(validation_milestones, validation_milestones[1:])
        ), "Validation milestones must be increasing."

        self.training_time = training_time
        self.validation_milestones = validation_milestones or []
        self.callbacks = CallbackList(callbacks or [])

    def __call__(
        self,
        net: nn.HybridBlock,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        logger.info("Start model training")
        net.initialize(ctx=self.ctx, init=self.init)

        with HybridContext(
            net=net,
            hybridize=self.hybridize,
            static_alloc=True,
            static_shape=True,
        ):
            self._train_loop(net, train_iter, validation_iter)

    def _train_loop(  # pylint: disable=too-many-statements
        self,
        net: nn.HybridBlock,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader],
    ) -> None:
        optimizer = mx.optimizer.Adam(
            learning_rate=self.learning_rate,
            wd=self.weight_decay,
            clip_gradient=self.clip_gradient,
        )

        trainer = mx.gluon.Trainer(
            net.collect_params(),
            optimizer=optimizer,
            kvstore="device",
        )

        first_forward = True
        time_elapsed = 0
        validation_idx = 0

        def loop(
            batch_iter: DataLoader,
            num_batches_to_use: Optional[int] = None,
            is_training: bool = True,
        ) -> mx.metric.Loss:
            nonlocal first_forward, time_elapsed, validation_idx

            tic = time.time()
            subtic = 0

            epoch_loss = mx.metric.Loss()
            batch_iter = itertools.islice(batch_iter, num_batches_to_use)

            it = tqdm(batch_iter, total=num_batches_to_use)
            for batch_no, batch in enumerate(it, start=1):
                # `batch` here is expected to be a dictionary whose fields
                # should correspond 1-to-1 with the network inputs
                # see below how `batch.values()` is fed into the network
                if first_forward:
                    tictic = time.time()
                    first_forward = False
                    _ = net(*batch.values())
                    self.callbacks.on_network_initialization_end(net)
                    subtic += time.time() - tictic

                with mx.autograd.record():  # type: ignore
                    # we set the mode explicitly as by default mxnet assumes
                    # predict mode and hence dropout layers are not used if
                    # the mode is not explicitly set to training
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

                # pylint: disable=no-member
                if not np.isfinite(ndarray.sum(loss).asscalar()):  # type: ignore
                    logger.warning(
                        "Batch [%d] gave NaN loss and it will be ignored",
                        batch_no,
                    )
                else:
                    if is_training:
                        loss.backward()
                        trainer.step(batch_size)
                    epoch_loss.update(None, preds=loss)

                if is_training:
                    total_time_elapsed = (
                        time_elapsed + time.time() - tic - subtic
                    )

                    orig_lr = trainer.learning_rate
                    tictic = time.time()
                    self.callbacks.on_train_batch_end(net, total_time_elapsed)
                    subtic += time.time() - tictic
                    if trainer.learning_rate != orig_lr:
                        logger.info(
                            "Trainer learning rate set to %f",
                            trainer.learning_rate,
                        )

                lv = _loss_value(epoch_loss)
                it.set_postfix(
                    ordered_dict={
                        ("" if is_training else "validation_")
                        + "avg_epoch_loss": lv,
                    },
                    refresh=False,
                )

                # Check if should finish
                if is_training:
                    if total_time_elapsed > self.training_time:  # type: ignore
                        time_elapsed = total_time_elapsed  # type: ignore
                        break
                    if len(self.validation_milestones) > validation_idx and (
                        total_time_elapsed  # type: ignore
                        > self.validation_milestones[validation_idx]
                    ):
                        time_elapsed = total_time_elapsed  # type: ignore
                        validation_idx += 1
                        break
                # If validating, call the callback with the loss
                else:
                    self.callbacks.on_validation_epoch_end(lv)

            # mark epoch end time and log time cost of current epoch
            toc = time.time()
            logger.info("Elapsed time %.3f seconds", toc - tic)
            logger.info(
                "Evaluation metric '%s'=%f",
                ("" if is_training else "validation_") + "epoch_loss",
                lv,  # type: ignore
            )

            return epoch_loss

        self.callbacks.on_train_start(trainer)
        while True:
            loop(train_iter)
            if validation_iter is not None:
                loop(validation_iter, is_training=False)
            if time_elapsed > self.training_time:
                break

        logger.info("End model training")


def _loss_value(loss: mx.metric.Loss) -> float:
    return loss.get_name_value()[0][1]
