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


from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Tuple, List, Iterator
from pathlib import Path
from functools import partial
from copy import deepcopy
import math

import torch as pt
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .callback import Callback, CallbackList, Optimization, DebugStopper
from .distributed import is_distributed, synchronize, is_main_process

if TYPE_CHECKING:
    from ..dataset import MetaDataset
    from ..metrics import MeterDict


class Trainer(object):
    def __init__(
        self,
        dataset: MetaDataset,
        model: nn.Module,
        optimizer: partial,
        scheduler: Optional[partial],
        metrics: MeterDict,
        callbacks: List[Callback],
        log_dir: Union[Path, str],
        n_epochs: int,
        nb_epoch: Optional[int],
        batch_size: Union[int, Tuple[int, int]],
        n_loader_workers: int = 0,
        cuda_device: int = 0,
        debug: bool = False,
    ) -> None:
        self.dataset = dataset
        self.cuda_device = cuda_device if pt.cuda.is_available() else -1
        if self.cuda_device >= 0:
            pt.cuda.set_device(self.cuda_device)
            self.model = model.cuda(self.cuda_device)
        else:
            self.model = model.cpu()
        if is_distributed():
            device_ids = None if self.cuda_device < 0 else [self.cuda_device]
            self.model = DistributedDataParallel(
                self.model, device_ids=device_ids
            )
        self.optimizer = optimizer(self.model.parameters())
        if scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler(self.optimizer)
        self.metrics = metrics
        self.register_callback(*callbacks)
        self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        self.n_epochs = n_epochs
        self.nb_epoch = nb_epoch
        if isinstance(batch_size, int):
            self.batch_size = self.eval_batch_size = batch_size
        else:
            self.batch_size, self.eval_batch_size = batch_size
        self.n_loader_workers = n_loader_workers
        self.debug = debug

        self.global_step = 0
        self._signal_exit = False
        self._signal_break = False
        self._signal_valid_break = False

    def register_callback(self, *callbacks: Callback):
        if hasattr(self, "callbacks"):
            for callback in callbacks:
                self.callbacks.append(callback)
        else:
            callbacks = list(callbacks)
            has_optimization = False
            has_debug_stopper = False
            for callback in callbacks:
                if isinstance(callback, Optimization):
                    has_optimization = True
                elif isinstance(callback, DebugStopper):
                    has_debug_stopper = True
            if not has_optimization:
                callbacks.insert(0, Optimization())
            if not has_debug_stopper:
                callbacks.insert(0, DebugStopper())
            self.callbacks = CallbackList(*callbacks, trainer=self)

    def get_model_core(self) -> nn.Module:
        if is_distributed():
            return self.model.module
        else:
            return self.model

    def dump(
        self,
        tag: str,
        save_model: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_metrics: bool = True,
    ) -> None:
        if is_main_process():
            # in distributed training, only model from process 0 is dumped
            state = {"global_step": self.global_step}
            if save_model:
                state["model"] = self.get_model_core().state_dict()
            if save_optimizer:
                state["optimizer"] = self.optimizer.state_dict()
            if save_scheduler and (self.scheduler is not None):
                state["scheduler"] = self.scheduler.state_dict()
            if save_metrics:
                state["metrics"] = self.metrics.state_dict()
            pt.save(state, self.log_dir.joinpath(f"{tag}.pt.tar"))
        # make sure non-base processes wait until process 0 finish checkpointing
        synchronize()

    def load(
        self,
        tag: str,
        load_model: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_metrics: bool = True,
    ) -> None:
        try:
            path = self.log_dir.joinpath(f"{tag}.pt.tar")
            state = pt.load(
                path,
                map_location=f"cuda:{self.cuda_device}"
                if self.cuda_device >= 0
                else "cpu",
            )
            print(f"Load checkpoint from {path}")
        except FileNotFoundError:
            raise ValueError(f"invalid tag {tag}")
        err_msg = "the checkpoint does not have {} data."
        global_step = state.get("global_step")
        if global_step is None:
            raise KeyError(err_msg.format("global_step"))
        self.global_step = global_step
        if load_model:
            module_state = state.get("model")
            if module_state is None:
                raise KeyError(err_msg.format("model"))
            self.get_model_core().load_state_dict(module_state)
        if load_optimizer:
            optimizer_state = state.get("optimizer")
            if optimizer_state is None:
                raise KeyError(err_msg.format("optimization"))
            self.optimizer.load_state_dict(optimizer_state)
        if load_scheduler and self.scheduler is not None:
            scheduler_state = state.get("scheduler")
            if scheduler_state is None:
                raise KeyError(err_msg.format("scheduler"))
            self.scheduler.load_state_dict(scheduler_state)
        if load_metrics:
            metrics_state = state.get("metrics")
            if metrics_state is None:
                raise KeyError(err_msg.format("metrics"))
            self.metrics.load_state_dict(metrics_state)

    @property
    def train_loader(self) -> Iterator:
        return self.dataset.train_loader(
            batch_size=self.batch_size,
            shuffle=(not self.debug),
            cuda_device=self.cuda_device,
            n_workers=self.n_loader_workers,
            n_batches=self.nb_epoch,
        )

    @property
    def valid_loader(self) -> Iterator:
        return self.dataset.valid_loader(
            batch_size=self.eval_batch_size,
            cuda_device=self.cuda_device,
            n_workers=self.n_loader_workers,
        )

    def fit(self) -> None:
        self.callbacks.on_train_begin()
        try:
            self._epoch_loop()
        except KeyboardInterrupt:
            if not self.debug:
                if input("Do you want to save the current model? (Y/n)") in [
                    "Y",
                    "y",
                ]:
                    self.dump(tag=f"epoch{epoch+1:02d}_interrupt")
            return
        finally:
            self.callbacks.on_train_end()

    def _epoch_loop(self) -> None:
        for epoch in range(self.n_epochs):
            self.model.train()
            self.callbacks.on_epoch_begin(epoch)
            self._batch_loop(epoch)
            self.evaluate(epoch)
            self.callbacks.on_epoch_end(epoch)
            self.metrics.restart()
            if self._signal_exit:
                break

    def _batch_loop(self, epoch: int) -> None:
        for batch, data in enumerate(self.train_loader):
            self.callbacks.on_batch_begin(batch)
            self._train(*data)
            self.callbacks.on_batch_end(batch)
            if self._signal_break:
                break

    def _train(self, *data):
        raise NotImplementedError

    def evaluate(self, epoch: int) -> None:
        _ = self.model.eval()
        with pt.no_grad():
            self.callbacks.on_valid_begin(epoch)
            self._valid_batch_loop(epoch)
            self.callbacks.on_valid_end(epoch)
        synchronize()

    def _valid_batch_loop(self, epoch: int) -> None:
        for batch, data in enumerate(self.valid_loader):
            self.callbacks.on_valid_batch_begin(batch)
            self._validate(*data)
            self.callbacks.on_valid_batch_end(batch)
            if self._signal_valid_break:
                break

    def _validate(self, *data):
        raise NotImplementedError
