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


from typing import Callable, Optional, List, Tuple, Dict, Union
from functools import partial
from itertools import chain
from pathlib import Path

from torch import Tensor, BoolTensor
from torch.optim import AdamW
import numpy as np

from tslib.dataset import MetaDataset
from tslib.engine import Trainer, Evaluator, HyperOptManager
from tslib.engine.distributed import (
    reduce_value,
    is_main_process,
)
from tslib.engine.callback import *
from tslib.metrics import (
    NumericalAverageMeter,
    BatchAverageMeter,
    MeanDeviationMeter,
    MeterDict,
)
from ..estimator import AttentionEstimator


class AttentionTrainer(Trainer):
    def _batch_loop(self, epoch: int):
        for batch, data in enumerate(self.train_loader):
            self.callbacks.on_batch_begin(batch)
            self._train(*data)
            self.callbacks.on_batch_end(
                batch,
            )
            if self._signal_break:
                break

    def _train(self, *data):
        loss = self.model(*data)
        loss.mean().backward()
        loss = reduce_value(loss, dst=None)
        self.metrics.train.loss.update(loss)
        self.metrics.train.bc_loss.update(self.model.bc_loss)
        self.metrics.train.fc_loss.update(self.model.fc_loss)

    def _validate(self, *data):
        _ = self.model(*data)
        self.metrics.valid.bc_loss.update(self.model.bc_loss)
        self.metrics.valid.fc_loss.update(self.model.fc_loss)
        self.metrics.valid.fc_ND.update(
            self.model.fc_loss, self.model.denominator
        )

    @classmethod
    def from_configs(
        cls,
        dataset: MetaDataset,
        model: AttentionEstimator,
        log_dir: Union[Path, str],
        cuda_device: int,
        n_epochs: int,
        nb_epoch: Optional[int] = None,
        batch_size: Union[int, Tuple[int, int]] = 100,
        max_grad_norm: float = 1e0,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        optimizer = partial(
            AdamW,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        meters = MeterDict(
            meterdicts={
                "train": MeterDict(
                    meters=dict(
                        loss=BatchAverageMeter(),
                        grad_norm=NumericalAverageMeter(),
                        fc_loss=BatchAverageMeter(),
                        bc_loss=BatchAverageMeter(),
                    )
                ),
                "valid": MeterDict(
                    meters=dict(
                        fc_loss=BatchAverageMeter(),
                        bc_loss=BatchAverageMeter(),
                        fc_ND=MeanDeviationMeter(),
                    )
                ),
                "test": MeterDict(
                    meters=dict(
                        fc_loss=BatchAverageMeter(),
                        bc_loss=BatchAverageMeter(),
                        fc_ND=MeanDeviationMeter(),
                    )
                ),
            }
        )

        callbacks = [
            Optimization(max_grad_norm=max_grad_norm),
            ProgressBar(
                batch_monitors=["train/loss", "train/grad_norm"],
                epoch_monitors=[
                    "train/fc_loss",
                    "valid/fc_loss",
                    "valid/fc_ND",
                ],
            ),
            Checkpoint(monitor="valid/fc_loss"),
            Tensorboard(
                batch_monitors=[
                    "train/loss",
                    "train/bc_loss",
                    "train/fc_loss",
                ],
                epoch_monitors=[
                    "valid/bc_loss",
                    "valid/fc_loss",
                    "valid/fc_ND",
                ],
            ),
        ]

        return cls(
            dataset,
            model,
            optimizer,
            None,
            metrics=meters,
            callbacks=callbacks,
            log_dir=log_dir,
            n_epochs=n_epochs,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            cuda_device=cuda_device,
            debug=debug,
        )


class AttentionEvaluator(Evaluator):
    def _test(self, *data):
        _ = self.model(*data)
        bc_loss = self.model.bc_loss
        fc_loss = self.model.fc_loss
        self.metrics.test.bc_loss.update(self.model.bc_loss)
        self.metrics.test.fc_loss.update(self.model.fc_loss)
        self.metrics.test.fc_ND.update(
            self.model.fc_loss, self.model.denominator
        )

    def _predict(self, *data) -> np.ndarray:
        _ = self.model(*data)
        truth = data[0].cpu().numpy()
        preds = self.model.forecast[..., 0].cpu().numpy()
        dummy = np.zeros(truth.shape[:-1] + preds.shape[-1:]) + np.nan
        preds = np.concatenate([dummy[:, : -preds.shape[1]], preds], axis=1)
        curve = np.concatenate([truth, preds], axis=-1)
        return curve

    @classmethod
    def from_trainer(
        cls,
        trainer: AttentionTrainer,
        model_tag: str = "best",
    ):
        return super(AttentionEvaluator, cls).from_trainer(trainer, model_tag)
