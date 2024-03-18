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
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class Learner:
    """
    The Learner class encapsulate the model that will be trained
    as well as the optimizer, the loss function and the data. It does not do
    anything specific but makes the whole code neater.
    """

    def __init__(self, model, opt, data, loss_func=None):
        self.model = model
        self.opt = opt
        self.data = data
        self.loss_func = loss_func

    def __repr__(self):
        rep = (
            "Learner: \n" + f"------| Model: \n{repr(self.model)} \n"
            f"------| Optimizer: \n{repr(self.opt)} \n"
            f"------| Data: \n{repr(self.data)}."
        )
        return rep


class Runner:
    """
    Runner is the main class to train the network. It is instantciated with
    a list of callbacks, that will specify how the training is done.
    To train the model, the function fit must be called. It takes as parameters
    a learner and a number of epochs. For each epoch, it will iterate over
    the dataloader and exectute such functions: self('name'). Exectuting such
    functions will iterate over all the callbacks specified in the cbs
    argument. If it finds a function called 'name' in the callback it will
    execute it.
    """

    def __repr__(self):
        rep = "\nRunner: List of Callbacks used :\n"
        for cb in sorted(self.cbs, key=lambda x: x._order):
            rep += f"--| {repr(cb)} \n"
        return rep

    def __init__(self, cbs: list):
        self.cbs = cbs
        self.stop = False

    @property
    def opt(self):
        return self.learn.opt

    @property
    def model(self):
        return self.learn.model

    @property
    def loss_func(self):
        return self.learn.loss_func

    @property
    def data(self):
        return self.learn.data

    def one_batch(self, item):
        try:
            self.item = item
            self("begin_batch")
            self("forward_pass")
            self("after_pred")
            self("compute_loss")
            self("after_loss")
            if not self.in_train:
                return
            self("backward")
            self("after_backward")
            self("step")
            self("after_step")
            self("zero_grad")

        except CancelBatchException:
            self("after_cancel_batch")

        finally:
            self("after_batch")

    def all_batches(self, dl):
        # self.n_iters = dl.length
        self.n_iters = len(dl)
        try:
            for itr, item in enumerate(dl):
                self.itr = itr
                self("before_batch")
                self.one_batch(item)

        except CancelEpochException:
            self("after_cancel_epoch")

    def fit(self, epochs, learn):
        self.epochs = epochs
        self.learn = learn
        try:
            for cb in self.cbs:
                cb.set_runner(self)
            self("begin_fit")

            for epoch in range(epochs):
                start = time.time()
                self.epoch = epoch
                if not self("begin_epoch"):
                    self.all_batches(self.data.train_dl)
                self("after_train")
                with torch.no_grad():
                    if not self("begin_validate"):
                        self.all_batches(self.data.valid_dl)
                    self("after_val")
                self("after_epoch")
                logger.info(
                    f"Epoch = {epoch}, Elapsed time {time.time() - start} seconds"
                )

                # current, peak = tracemalloc.get_traced_memory()
                # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
                # tracemalloc.stop()

        except CancelTrainException:
            self("after_cancel_train")

        finally:
            self("after_fit")
            # self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) or res
        return res


class Optimizers:
    """
    Class that encapsulates all the optimizers of the model, as the model
    contains multiple networks updated independently.
    """

    def __repr__(self):
        return (
            f"----------| Generator Optimizer : {repr(self.opt_generator)}\n"
            + f"----------| Discriminator Optimizer: {repr(self.opt_discriminator)}"
        )

    def __init__(
        self,
        opt_generator,
        opt_discriminator,
    ):
        self.opt_generator = opt_generator
        self.opt_discriminator = opt_discriminator

    def get_opt(self, net: str = "generator"):
        try:
            if net == "generator":
                return self.opt_generator
            elif net == "discriminator":
                return self.opt_discriminator
            else:
                raise CancelTrainException

        except CancelTrainException:
            logger.info(
                "Please specify a network to get\
        the relevant optimizer generator, discriminator."
            )

    def step(self, net: str = "generator"):
        self.get_opt(net).step()

    def zero_grad(self, net: str = "generator"):
        self.get_opt(net).zero_grad()

    def state_dict(self, net: str = "generator"):
        return self.get_opt(net).state_dict()
