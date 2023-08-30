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
from typing import TYPE_CHECKING, Optional, Iterable, Iterator
from pathlib import Path
import math

from tqdm import tqdm
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch as pt
import numpy as np

if TYPE_CHECKING:
    from ..dataset import ForecastingDataset, MetaDataset
    from ..metrics import MeterDict
    from .trainer import Trainer


class Evaluator(object):
    def __init__(
        self,
        dataset: MetaDataset,
        model: nn.Module,
        metrics: MeterDict,
        log_dir: Path,
        batch_size: int,
        n_loader_workers: int = 0,
        cuda_device: int = 0,
        debug: bool = False,
        model_tag: str = "best",
    ):
        self.dataset = dataset
        self.metrics = metrics
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.n_loader_workers = n_loader_workers
        self.cuda_device = cuda_device if pt.cuda.is_available() else -1
        self.pin_memory = self.cuda_device >= 0
        self.debug = debug
        if self.cuda_device >= 0:
            pt.cuda.set_device(self.cuda_device)
            self.model = model.cuda(self.cuda_device)
        else:
            self.model = model.cpu()
        self.load(model_tag)

    def load(self, tag: str):
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
        module_state = state.get("model")
        if module_state is None:
            raise KeyError("the checkpoint does not have model data.")
        self.model.load_state_dict(module_state)

    @property
    def test_loader(self) -> Iterator:
        return self.dataset.test_loader(
            batch_size=self.batch_size,
            cuda_device=self.cuda_device,
            n_workers=self.n_loader_workers,
        )

    def evaluate(self) -> None:
        device = "cpu" if self.cuda_device < 0 else f"cuda:{self.cuda_device}"
        prog_bar = tqdm(
            desc=f"({device}) test",
            total=int(math.ceil(self.dataset.test_size / self.batch_size)),
            unit="batch",
        )
        _ = self.model.eval()
        for batch, data in enumerate(self.test_loader):
            with pt.no_grad():
                self._test(*data)
            prog_bar.update(1)
            if self.debug and batch > 0:
                break
        prog_bar.close()

    def _test(self, *data) -> None:
        raise NotImplementedError

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
        preds = []
        for batch, data in enumerate(loader):
            with pt.no_grad():
                result = self._predict(*data)
            preds.append(result)
            prog_bar.update(1)
            if self.debug and batch > 0:
                break
        prog_bar.close()
        preds = np.concatenate(preds, axis=0)
        return preds

    def _predict(self, *data) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_trainer(cls, trainer: Trainer, model_tag: str):
        return cls(
            dataset=trainer.dataset,
            model=trainer.get_model_core(),
            metrics=trainer.metrics,
            log_dir=trainer.log_dir,
            batch_size=trainer.eval_batch_size,
            n_loader_workers=trainer.n_loader_workers,
            cuda_device=trainer.cuda_device,
            debug=trainer.debug,
            model_tag=model_tag,
        )

    @classmethod
    def from_dump(
        cls,
        dataset: MetaDataset,
        dump_dir: Union[str, Path],
        cuda_device: int,
        model_tag: str,
    ):
        raise NotImplementedError
