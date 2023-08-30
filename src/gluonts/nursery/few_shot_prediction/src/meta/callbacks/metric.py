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

from typing import Optional, Union, List
from pytorch_lightning import Callback, Trainer
import torch
from gluonts.time_feature import get_seasonality

from meta.models.module import MetaLightningModule
from meta.common.torch import tensor_to_np
from meta.metrics.numpy import compute_metrics


class QuantileMetricLoggerCallback(Callback):
    """
    A callback that computes additional metrics on a numpy representation of the dataset every n epochs.
    The computed values are logged to the output file of the pytorch lightning logger.

    Args:
        quantiles: The quantiles that are predicted.
        split: Specifies the split the batch comes from (i.e. from training or validation split)
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(
        self,
        quantiles: List[str],
        split: Optional[Union["train", "val"]] = None,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.split = split
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs:
            return
        dm_super = trainer.lightning_module.trainer.datamodule
        dm_val = dm_super.data_modules_val[0]
        dl = dm_val.val_dataloader()
        split = dm_val.splits.val()
        pred = []
        for batch in dl:
            batch = batch.to(pl_module.device)
            pred.append(
                pl_module.model(
                    supps=batch.support_set, query=batch.query_past
                )
            )

        # redo standardization for evaluation
        pred = split.data().rescale_dataset(torch.cat(pred, dim=0).cpu())
        pred = tensor_to_np(pred)

        # use only the length that should be included for evaluation
        pred = pred[:, : dm_val.prediction_length, ...]
        m = compute_metrics(
            pred,
            split.evaluation(),
            quantiles=self.quantiles,
            seasonality=get_seasonality(dm_val.meta.freq),
        )
        self.log("metrics", m)
