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
from typing import TYPE_CHECKING, Optional, List, Dict
from pathlib import Path
import math

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback(object):
    def __init__(self, trainer: Optional[Trainer] = None):
        if trainer is not None:
            self.set_trainer(trainer)
        else:
            self.trainer = None

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_epoch_end(self, epoch, **kwargs):
        pass

    def on_batch_begin(self, batch, **kwargs):
        pass

    def on_batch_end(self, batch, **kwargs):
        pass

    def on_valid_begin(self, epoch, **kwargs):
        pass

    def on_valid_end(self, epoch, **kwargs):
        pass

    def on_valid_batch_begin(self, batch, **kwargs):
        pass

    def on_valid_batch_end(self, batch, **kwargs):
        pass


class CallbackList(Callback):
    def __init__(self, *callbacks, trainer: Optional[Trainer] = None):
        self.callbacks = list(callbacks)
        super(CallbackList, self).__init__(trainer)

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def append(self, callback: Callback):
        if not isinstance(callback, Callback):
            raise ValueError("The given object is not a callback.")
        if isinstance(callback, CallbackList):
            raise ValueError("Cannot add a callback list to a callback list")
        callback.set_trainer(self.trainer)
        self.callbacks.append(callback)

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]

    def on_train_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_epoch_begin(self, epoch, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, **kwargs)

    def on_batch_begin(self, batch, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, **kwargs)

    def on_batch_end(self, batch, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(batch, **kwargs)

    def on_valid_begin(self, epoch, **kwargs):
        for callback in self.callbacks:
            callback.on_valid_begin(epoch, **kwargs)

    def on_valid_end(self, epoch, **kwargs):
        for callback in self.callbacks:
            callback.on_valid_end(epoch, **kwargs)

    def on_valid_batch_begin(self, batch, **kwargs):
        for callback in self.callbacks:
            callback.on_valid_batch_begin(batch, **kwargs)

    def on_valid_batch_end(self, batch, **kwargs):
        for callback in self.callbacks:
            callback.on_valid_batch_end(batch, **kwargs)


class Optimization(Callback):
    def __init__(
        self,
        grad_cum_step: int = 1,
        max_grad_norm: Optional[float] = None,
        lr_decay_freq: str = "batch",
    ):
        super(Optimization, self).__init__()
        if lr_decay_freq not in ["batch", "epoch"]:
            raise ValueError(
                f'learning rate decay frequency must be either "batch" or "epoch"'
            )
        self.grad_cum_step = grad_cum_step
        self.lr_decay_freq = lr_decay_freq
        self.max_grad_norm = max_grad_norm

    def on_train_begin(self, **kwargs):
        self.trainer.optimizer.zero_grad()
        self.grad_cum_counter = self.grad_cum_step

    def on_batch_end(self, batch, **kwargs):
        self.grad_cum_counter -= 1
        if self.grad_cum_counter == 0:
            if self.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(
                    self.trainer.model.parameters(), self.max_grad_norm
                )
                if "grad_norm" in self.trainer.metrics.train:
                    self.trainer.metrics.train.grad_norm.update(grad_norm)
            self.trainer.optimizer.step()
            self.trainer.global_step += 1
            if (
                self.lr_decay_freq == "batch"
                and self.trainer.scheduler is not None
            ):
                self.trainer.scheduler.step()
            self.trainer.optimizer.zero_grad()
            self.grad_cum_counter = self.grad_cum_step
        else:
            pass

    def on_epoch_end(self, epoch, **kwargs):
        if (
            self.lr_decay_freq == "epoch"
            and self.trainer.scheduler is not None
        ):
            self.trainer.scheduler.step()


class DebugStopper(Callback):
    def __init__(self, debug_stop_batch: int = 1, debug_stop_epoch: int = 1):
        super(DebugStopper, self).__init__()
        self.debug_stop_batch = debug_stop_batch
        self.debug_stop_epoch = debug_stop_epoch

    def on_train_begin(self, **kwargs):
        self.trainer._signal_exit = False
        self.trainer._signal_break = False
        self.trainer._signal_valid_break = False

    def on_batch_end(self, batch, **kwargs):
        if self.trainer.debug:
            if batch >= self.debug_stop_batch - 1:
                self.trainer._signal_break = True
        else:
            self.trainer._signal_break = False

    def on_epoch_end(self, epoch, **kwargs):
        if self.trainer.debug:
            if epoch >= self.debug_stop_epoch - 1:
                self.trainer._signal_exit = True
        else:
            self.trainer._signal_exit = False

    def on_valid_batch_end(self, batch, **kwargs):
        if self.trainer.debug:
            if batch >= self.debug_stop_batch - 1:
                self.trainer._signal_valid_break = True
        else:
            self.trainer._signal_valid_break = False


class EarlyStopping(Callback):
    def __init__(
        self,
        patience: int,
        monitor: str = "valid/loss",
        minimize: bool = True,
        threshold: float = 0.0,
    ):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.monitor = monitor
        self.minimize = minimize
        self.threshold = threshold

    def on_train_begin(self, **kwargs):
        self.wait = 0
        self._stopped_epoch = 0
        self.best = float("inf")
        if not self.minimize:
            self.best = -self.best

    def on_epoch_end(self, epoch, **kwargs):
        current_value = self.trainer.metrics[self.monitor].value
        diff = current_value - self.best
        better = (
            (diff < -self.threshold)
            if self.minimize
            else (diff > self.threshold)
        )
        if better:
            self.best = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait == self.patience:
                self.trainer._signal_exit = True
                self._stopped_epoch = epoch + 1

    def on_train_end(self, **kwargs):
        if self._stopped_epoch > 0:
            print(f"Early stopping at epoch {self._stopped_epoch:02d}")


class Checkpoint(Callback):
    def __init__(
        self,
        monitor: str = "valid/loss",
        minimize: bool = True,
        threshold: float = 0.0,
        keep_epoch_dump: bool = True,
    ):
        super(Checkpoint, self).__init__()
        self.monitor = monitor
        self.minimize = minimize
        self.threshold = threshold
        self.keep_epoch_dump = keep_epoch_dump

    def on_train_begin(self, **kwargs):
        self.best = float("inf") if self.minimize else float("-inf")

    def on_epoch_end(self, epoch, **kwargs):
        current_value = self.trainer.metrics[self.monitor].value
        diff = current_value - self.best
        better = (
            (diff < -self.threshold)
            if self.minimize
            else (diff > self.threshold)
        )
        if better:
            self.best = current_value
            self.trainer.dump(tag="best")
        if self.keep_epoch_dump:
            self.trainer.dump(tag=f"epoch{epoch+1:02d}")


class ProgressBar(Callback):
    def __init__(
        self,
        batch_monitors: Optional[List[str]] = None,
        epoch_monitors: Optional[List[str]] = None,
    ):
        super(ProgressBar, self).__init__()
        self.batch_monitors = batch_monitors or []
        self.epoch_monitors = epoch_monitors or []

    def on_epoch_begin(self, epoch: int, **kwargs):
        if self.trainer.nb_epoch is None:
            n_batch = int(
                math.ceil(
                    self.trainer.dataset.train_size / self.trainer.batch_size
                )
            )
        else:
            n_batch = self.trainer.nb_epoch
        device = (
            "cpu"
            if self.trainer.cuda_device < 0
            else f"cuda:{self.trainer.cuda_device}"
        )
        self.prog_bar = tqdm(
            desc=f"({device}) {epoch+1}/{self.trainer.n_epochs}",
            total=n_batch,
            unit="batch",
        )

    def on_batch_end(self, batch, **kwargs):
        self.prog_bar.update(1)
        postfix = {}
        for k in self.batch_monitors:
            meter = self.trainer.metrics.get(k)
            if meter is None:
                v = kwargs[k]
            else:
                v = meter.value
            postfix[k] = v
        self.prog_bar.set_postfix(postfix)

    def on_valid_begin(self, epoch, **kwargs):
        n_batch = int(
            math.ceil(
                self.trainer.dataset.valid_size / self.trainer.eval_batch_size
            )
        )
        device = (
            "cpu"
            if self.trainer.cuda_device < 0
            else f"cuda:{self.trainer.cuda_device}"
        )
        self.sub_prog_bar = tqdm(
            desc=f"    ({device}) validating epoch {epoch+1}",
            total=n_batch,
            unit="batch",
            position=1,
            leave=False,
        )

    def on_valid_batch_end(self, batch, **kwargs):
        self.sub_prog_bar.update(1)

    def on_valid_end(self, epoch, **kwargs):
        self.sub_prog_bar.close()

    def on_epoch_end(self, epoch, **kwargs):
        postfix = {}
        for k in self.epoch_monitors:
            meter = self.trainer.metrics.get(k)
            if meter is None:
                v = kwargs[k]
            else:
                v = meter.value
            postfix[k] = v
        self.prog_bar.set_postfix(postfix)
        self.prog_bar.close()


class Tensorboard(Callback):
    def __init__(
        self,
        batch_monitors: Optional[List[str]] = None,
        epoch_monitors: Optional[List[str]] = None,
        hparams: Optional[Dict] = None,
        metrics: Optional[List[str]] = None,
        comment: str = "",
    ):
        super(Tensorboard, self).__init__()
        self.comment = comment
        self.batch_monitors = batch_monitors or []
        self.epoch_monitors = epoch_monitors or []
        self.hparams = hparams
        self.metrics = metrics

    def on_train_begin(self, **kwargs):
        self.writer = SummaryWriter(self.trainer.log_dir, self.comment)

    def on_batch_end(self, batch, **kwargs):
        for k in self.batch_monitors:
            v = kwargs.get(k)
            if v is None:
                v = self.trainer.metrics[k].cache
            self.writer.add_scalar(k, v, global_step=self.trainer.global_step)

    def on_epoch_end(self, epoch, **kwargs):
        for k in self.epoch_monitors:
            v = kwargs.get(k)
            if v is None:
                v = self.trainer.metrics[k].value
            self.writer.add_scalar(k, v, global_step=self.trainer.global_step)
        weights = kwargs.get("weights", {})
        for k, v in weights.items():
            self.writer.add_histogram(
                f"weights/{k}", v, global_step=self.trainer.global_step
            )
        grads = kwargs.get("grads", {})
        for k, v in grads.items():
            self.writer.add_histogram(
                f"grads/{k}", v, global_step=self.trainer.global_step
            )

    def on_train_end(self, **kwargs):
        if self.hparams is not None:
            if self.metrics is None:
                metrics = self.metrics.valid.best
            else:
                metrics = {
                    k: self.trainer.metrics[k].best for k in self.metrics
                }
            self.writer.add_hparams(self.hparams, metrics)
        self.writer.close()
