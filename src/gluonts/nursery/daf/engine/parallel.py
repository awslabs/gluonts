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
from torch.utils.data import Dataset
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
from ..estimator import (
    DomAdaptEstimator,
    AdversarialDomAdaptEstimator,
)


class DomAdaptTrainer(Trainer):
    def _train(self, *data):
        loss = self.model(*data)
        loss.mean().backward()
        loss = reduce_value(loss, dst=None)
        self.metrics.train.loss.update(loss)
        self.metrics.train.src.bc_loss.update(self.model.src.bc_loss)
        self.metrics.train.src.fc_loss.update(self.model.src.fc_loss)
        self.metrics.train.tgt.bc_loss.update(self.model.tgt.bc_loss)
        self.metrics.train.tgt.fc_loss.update(self.model.tgt.fc_loss)

    def _validate(self, *data):
        _ = self.model(*data)
        self.metrics.valid.src.bc_loss.update(self.model.src.bc_loss)
        self.metrics.valid.src.fc_loss.update(self.model.src.fc_loss)
        self.metrics.valid.src.fc_ND.update(
            self.model.src.fc_loss, self.model.src.denominator
        )
        self.metrics.valid.tgt.bc_loss.update(self.model.tgt.bc_loss)
        self.metrics.valid.tgt.fc_loss.update(self.model.tgt.fc_loss)
        self.metrics.valid.tgt.fc_ND.update(
            self.model.tgt.fc_loss, self.model.tgt.denominator
        )

    @classmethod
    def from_configs(
        cls,
        dataset: MetaDataset,
        model: DomAdaptEstimator,
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
                    meters=dict(loss=BatchAverageMeter()),
                    meterdicts=dict(
                        src=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                            )
                        ),
                        tgt=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                            )
                        ),
                    ),
                ),
                "valid": MeterDict(
                    meterdicts=dict(
                        src=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                                fc_ND=MeanDeviationMeter(),
                            )
                        ),
                        tgt=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                                fc_ND=MeanDeviationMeter(),
                            )
                        ),
                    ),
                ),
                "test": MeterDict(
                    meterdicts=dict(
                        src=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                                fc_ND=MeanDeviationMeter(),
                            )
                        ),
                        tgt=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                                fc_ND=MeanDeviationMeter(),
                            )
                        ),
                    ),
                ),
            }
        )

        callbacks = [
            Optimization(max_grad_norm=max_grad_norm),
            ProgressBar(
                batch_monitors=[
                    "train/loss",
                    "train/src/fc_loss",
                    "train/tgt/fc_loss",
                ],
                epoch_monitors=[
                    "train/src/fc_loss",
                    "train/tgt/fc_loss",
                    "valid/src/fc_loss",
                    "valid/tgt/fc_loss",
                    "valid/src/fc_ND",
                    "valid/tgt/fc_ND",
                ],
            ),
            Checkpoint(monitor="valid/tgt/fc_loss"),
            Tensorboard(
                batch_monitors=[
                    "train/loss",
                    "train/src/bc_loss",
                    "train/src/fc_loss",
                    "train/tgt/bc_loss",
                    "train/tgt/fc_loss",
                ],
                epoch_monitors=[
                    "valid/src/bc_loss",
                    "valid/src/fc_loss",
                    "valid/src/fc_ND",
                    "valid/tgt/bc_loss",
                    "valid/tgt/fc_loss",
                    "valid/tgt/fc_ND",
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


class DomAdaptEvaluator(Evaluator):
    def _test(self, *data):
        _ = self.model(*data)
        self.metrics.test.src.bc_loss.update(self.model.src.bc_loss)
        self.metrics.test.src.fc_loss.update(self.model.src.fc_loss)
        self.metrics.test.src.fc_ND.update(
            self.model.src.fc_loss, self.model.src.denominator
        )
        self.metrics.test.tgt.bc_loss.update(self.model.tgt.bc_loss)
        self.metrics.test.tgt.fc_loss.update(self.model.tgt.fc_loss)
        self.metrics.test.tgt.fc_ND.update(
            self.model.tgt.fc_loss, self.model.tgt.denominator
        )

    def predict(self, dataset: Optional[Dataset] = None) -> np.ndarray:
        if dataset is None:
            loader = self.test_loader
            data_size = self.dataset.test_size
        else:
            loader = MetaDataset._data_loader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False,
                cuda_device=self.cuda_device,
                is_training=False,
                n_workers=self.n_loader_workers,
                n_batches=None,
            )
            data_size = len(dataset)
        device = "cpu" if self.cuda_device < 0 else f"cuda:{self.cuda_device}"
        prog_bar = tqdm(
            desc=f"({device}) test",
            total=int(math.ceil(data_size / self.batch_size)),
            unit="batch",
        )
        _ = self.model.eval()
        src_preds, tgt_preds = [], []
        for batch, data in enumerate(loader):
            with pt.no_grad():
                src_result, tgt_result = self._predict(*data)
            src_preds.append(src_result)
            tgt_preds.append(tgt_result)
            prog_bar.update(1)
            if self.debug and batch > 0:
                break
        prog_bar.close()
        src_preds = np.concatenate(src_preds, axis=0)
        tgt_preds = np.concatenate(tgt_preds, axis=0)
        return src_preds, tgt_preds

    def _predict(self, *data) -> np.ndarray:
        _ = self.model(*data)
        src_truth = data[0].cpu().numpy()
        tgt_truth = data[1].cpu().numpy()
        src_preds = self.model.src.forecast[..., 0].cpu().numpy()
        tgt_preds = self.model.tgt.forecast[..., 0].cpu().numpy()
        src_dummy = (
            np.zeros(src_truth.shape[:-1] + src_preds.shape[-1:]) + np.nan
        )
        src_preds = np.concatenate(
            [src_dummy[:, : -src_preds.shape[1]], src_preds], axis=1
        )
        src_curve = np.concatenate([src_truth, src_preds], axis=-1)
        tgt_dummy = (
            np.zeros(tgt_truth.shape[:-1] + tgt_preds.shape[-1:]) + np.nan
        )
        tgt_preds = np.concatenate(
            [tgt_dummy[:, : -tgt_preds.shape[1]], tgt_preds], axis=1
        )
        tgt_curve = np.concatenate([tgt_truth, tgt_preds], axis=-1)
        return src_curve, tgt_curve

    @classmethod
    def from_trainer(
        cls,
        trainer: DomAdaptTrainer,
        model_tag: str = "best",
    ):
        return super(DomAdaptEvaluator, cls).from_trainer(trainer, model_tag)


class AdversarialDomAdaptTrainer(DomAdaptTrainer):
    def __init__(
        self,
        dataset: MetaDataset,
        model: AdversarialDomAdaptEstimator,
        optimizer: partial,
        scheduler: Optional[partial],
        **kwargs,
    ) -> None:
        super(AdversarialDomAdaptTrainer, self).__init__(
            dataset,
            model,
            optimizer,
            scheduler,
            **kwargs,
        )
        del self.optimizer
        self.optimizer = optimizer(
            [
                {
                    "params": chain(
                        self.model.src.generative_parameters(),
                        self.model.tgt.generative_parameters(),
                    )
                },
                {
                    "params": chain(
                        self.model.src.discriminative_parameters(),
                        self.model.tgt.discriminative_parameters(),
                    )
                },
            ]
        )

    def _train(self, *data):
        self.model.generative()
        gen_loss = self.model(*data)
        gen_loss.mean().backward()
        gen_loss = reduce_value(gen_loss, dst=None)
        self.metrics.train.loss.update(gen_loss)
        self.metrics.train.src.bc_loss.update(self.model.src.bc_loss)
        self.metrics.train.src.fc_loss.update(self.model.src.fc_loss)
        self.metrics.train.tgt.bc_loss.update(self.model.tgt.bc_loss)
        self.metrics.train.tgt.fc_loss.update(self.model.tgt.fc_loss)

        self.model.discriminative()
        disc_loss = self.model(*data)
        disc_loss.mean().backward()
        disc_loss = reduce_value(disc_loss, dst=None)
        self.metrics.train.disc.update(disc_loss)

    def _validate(self, *data):
        self.model.generative()
        _ = self.model(*data)
        self.metrics.valid.src.bc_loss.update(self.model.src.bc_loss)
        self.metrics.valid.src.fc_loss.update(self.model.src.fc_loss)
        self.metrics.valid.tgt.bc_loss.update(self.model.tgt.bc_loss)
        self.metrics.valid.tgt.fc_loss.update(self.model.tgt.fc_loss)

    @classmethod
    def from_configs(
        cls,
        dataset: MetaDataset,
        model: AdversarialDomAdaptEstimator,
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
                        disc=BatchAverageMeter(),
                        grad_norm=NumericalAverageMeter(),
                    ),
                    meterdicts=dict(
                        src=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                            )
                        ),
                        tgt=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                            )
                        ),
                    ),
                ),
                "valid": MeterDict(
                    meterdicts=dict(
                        src=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                                fc_ND=MeanDeviationMeter(),
                            )
                        ),
                        tgt=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                                fc_ND=MeanDeviationMeter(),
                            )
                        ),
                    ),
                ),
                "test": MeterDict(
                    meterdicts=dict(
                        src=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                                fc_ND=MeanDeviationMeter(),
                            )
                        ),
                        tgt=MeterDict(
                            meters=dict(
                                bc_loss=BatchAverageMeter(),
                                fc_loss=BatchAverageMeter(),
                                fc_ND=MeanDeviationMeter(),
                            )
                        ),
                    ),
                ),
            }
        )

        callbacks = [
            Optimization(max_grad_norm=max_grad_norm),
            ProgressBar(
                batch_monitors=[
                    "train/loss",
                    "train/disc",
                    "train/src/fc_loss",
                    "train/tgt/fc_loss",
                ],
                epoch_monitors=[
                    "train/src/fc_loss",
                    "train/tgt/fc_loss",
                    "valid/src/fc_loss",
                    "valid/tgt/fc_loss",
                    "valid/src/fc_ND",
                    "valid/tgt/fc_ND",
                ],
            ),
            Checkpoint(monitor="valid/tgt/fc_loss"),
            Tensorboard(
                batch_monitors=[
                    "train/loss",
                    "train/disc",
                    "train/src/bc_loss",
                    "train/src/fc_loss",
                    "train/tgt/bc_loss",
                    "train/tgt/fc_loss",
                ],
                epoch_monitors=[
                    "valid/src/bc_loss",
                    "valid/src/fc_loss",
                    "valid/src/fc_ND",
                    "valid/tgt/bc_loss",
                    "valid/tgt/fc_loss",
                    "valid/tgt/fc_ND",
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


class AdversarialDomAdaptEvaluator(DomAdaptEvaluator):
    @classmethod
    def from_trainer(
        cls,
        trainer: AdversarialDomAdaptTrainer,
        model_tag: str = "best",
    ):
        return super(AdversarialDomAdaptEvaluator, cls).from_trainer(
            trainer, model_tag
        )
