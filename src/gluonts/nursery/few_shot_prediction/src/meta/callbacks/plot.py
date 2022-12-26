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
import matplotlib.pyplot as plt

from meta.data.batch import TripletBatch
from meta.models.module import MetaLightningModule
from .common import get_save_dir_from_csvlogger, get_loss_steps
from meta.data.batch import SeriesBatch
from meta.vis.forecast import (
    plot_quantile_forecast,
    plot_forecast_supportset_attention,
)


class ForecastPlotLoggerCallback(Callback):
    """
    A callback that stores plots of the  predictions for a collection of samples every n epochs.
    The plots display the query (past and future) and forecasted quantiles.
    This callback is intended for models without support set and attention mechanism.

    Args:
        log_batch: For each sample in the batch the prediction is plotted when the callback is called.
            The batch size should not be too large (i.e. < 10).
        quantiles: The quantiles that are predicted.
        split: Specifies the split the batch comes from (i.e. from training or validation split)
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(
        self,
        log_batch: TripletBatch,
        quantiles: List[str],
        split: Optional[Union["train", "val"]] = None,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.log_batch = log_batch
        self.quantiles = quantiles
        self.split = split
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs:
            return
        # Bring the tensors to CPU
        log_batch = self.log_batch.to(device=pl_module.device)
        query_past = log_batch.query_past
        query_future = log_batch.query_future
        support_set = log_batch.support_set

        # Get model prediction
        pred = pl_module.model(supps=support_set, query=query_past)
        pred = SeriesBatch(
            pred, query_future.lengths, query_future.split_sections
        ).unpad(to_numpy=True, squeeze=True)
        qp = query_past.unpad(to_numpy=True, squeeze=True)
        qf = query_future.unpad(to_numpy=True, squeeze=True)
        fig = plot_quantile_forecast(qp, qf, pred, quantiles=self.quantiles)
        fig.savefig(
            get_save_dir_from_csvlogger(trainer.logger)
            / f"pred_{self.split + '_' if self.split else ''}ep{trainer.current_epoch}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


class ForecastSupportSetAttentionPlotLoggerCallback(Callback):
    """
    A callback that stores plots of the  predictions for a collection of samples every n epochs.
    The plots display the query (past and future), forecasted quantiles and the time series in the support set
    of this sample aligned with their attention scores. This callback works only for models with attention mechanism!

    Args:
        log_batch: For each sample in the batch the prediction is plotted when the callback is called.
            The batch size should not be too large (i.e. < 10).
        quantiles: The quantiles that are predicted.
        split: Specifies the split the batch comes from (i.e. from training or validation split)
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(
        self,
        log_batch: TripletBatch,
        quantiles: List[str],
        split: Optional[Union["train", "val"]] = None,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.log_batch = log_batch
        self.quantiles = quantiles
        self.split = split
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs:
            return
        # Bring the tensors to CPU
        log_batch = self.log_batch.to(device=pl_module.device)
        query_past = log_batch.query_past
        query_future = log_batch.query_future
        support_set = log_batch.support_set

        # Get model prediction
        pred, attention = pl_module.model(
            supps=support_set,
            query=query_past,
            return_attention=True,
        )
        pred = SeriesBatch(
            pred, query_future.lengths, query_future.split_sections
        ).unpad(to_numpy=True, squeeze=True)
        qp = query_past.unpad(to_numpy=True, squeeze=True)
        qf = query_future.unpad(to_numpy=True, squeeze=True)
        supps = support_set.unpad(to_numpy=True, squeeze=False)
        n_supps = support_set.split_sections[0]
        attention = attention.reshape(n_supps * attention.size()[0], 1, -1)
        attention = SeriesBatch(
            attention.transpose(1, 2),
            support_set.lengths,
            support_set.split_sections,
        ).unpad(to_numpy=True, squeeze=False)
        # attention = [[tensor_to_np(att) for att in supps] for supps in attention]

        fig = plot_forecast_supportset_attention(
            qp,
            qf,
            pred,
            supps=supps,
            attention=attention,
            quantiles=self.quantiles,
        )
        fig.savefig(
            get_save_dir_from_csvlogger(trainer.logger)
            / f"pred_supps_{self.split + '_' if self.split else ''}ep{trainer.current_epoch}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


class LossPlotLoggerCallback(Callback):
    """
    A callback that stores plots of the training and macro-averaged validation loss curve every n epochs.

    Args:
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(self, every_n_epochs: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        if (trainer.current_epoch < 1) or (
            trainer.current_epoch % self.every_n_epochs
        ):
            return
        self.plot_loss(trainer=trainer, pl_module=pl_module)

    def on_fit_end(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        if trainer.current_epoch < 1:
            return
        self.plot_loss(trainer=trainer, pl_module=pl_module)

    def plot_loss(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        train_loss, train_steps = get_loss_steps("train_loss", trainer)
        val_loss_macro, val_steps = get_loss_steps("val_loss_macro", trainer)

        fig, ax = plt.subplots()
        ax.semilogy(train_steps, train_loss, label="train_loss", alpha=0.5)
        ax.semilogy(
            val_steps, val_loss_macro, label="val_loss_macro", alpha=0.5
        )
        ax2 = ax.twinx()
        ax2.set_ylabel("crps scores", color="red")
        if pl_module.val_crps_scaled_macro:
            val_crps_scaled_macro, _ = get_loss_steps(
                "val_crps_scaled_macro", trainer
            )
            ax2.semilogy(
                val_steps,
                val_crps_scaled_macro,
                "r--",
                label="val_crps_scaled_macro",
            )
        else:
            val_crps_macro, _ = get_loss_steps("val_crps_macro", trainer)
            ax2.semilogy(
                val_steps, val_crps_macro, "r--", label="val_crps_macro"
            )
        ax2.legend(loc="upper right")

        ticks = range(0, trainer.current_epoch + 1, self.every_n_epochs)
        labels = [str(t) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(loc="upper left")

        fig.savefig(
            get_save_dir_from_csvlogger(trainer.logger) / f"loss.png",
            bbox_inches="tight",
        )
        plt.close(fig)


class CheatLossPlotLoggerCallback(LossPlotLoggerCallback):
    """
    A callback that stores plots of the training and multiple validation losses curve every n epochs.

    Args:
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
        dataset_names_val: The names of the datasets that the validation loss should be plotted for.
    """

    def __init__(self, dataset_names_val, **kwargs):
        super().__init__(**kwargs)
        self.dataset_names_val = dataset_names_val

    def plot_loss(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        train_loss, train_steps = get_loss_steps("train_loss", trainer)

        fig, ax = plt.subplots()
        ax.semilogy(train_steps, train_loss, label="train_loss", alpha=0.5)
        for d_name in self.dataset_names_val:
            val_loss, val_steps = get_loss_steps(f"{d_name}_val_loss", trainer)
            ax.semilogy(val_steps, val_loss, label=f"{d_name}_val_loss")

        ticks = range(0, trainer.current_epoch + 1, self.every_n_epochs)
        labels = [str(t) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(loc="upper left")

        fig.savefig(
            get_save_dir_from_csvlogger(trainer.logger) / f"loss.png",
            bbox_inches="tight",
        )
        plt.close(fig)


class MacroCRPSPlotCallback(Callback):
    """
    A callback that stores plots of the validation losses and a macro-averaged validation loss curve every n epochs.

    Args:
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        # We only compute this on the data as it is (no rescaling)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs:
            return
        self.plot_crps(trainer, scaled=False)
        dm_super = trainer.lightning_module.trainer.datamodule
        if dm_super.standardize:
            self.plot_crps(trainer, scaled=True)

    def on_fit_end(
        self, trainer: Trainer, pl_module: MetaLightningModule
    ) -> None:
        if trainer.current_epoch < 1:
            return
        self.plot_crps(trainer, scaled=False)
        dm_super = trainer.lightning_module.trainer.datamodule
        if dm_super.standardize:
            self.plot_crps(trainer, scaled=True)

    def plot_crps(self, trainer: Trainer, scaled: bool) -> None:
        cm = plt.get_cmap("tab20")
        suffix = "_scaled" if scaled else ""
        dm_super = trainer.lightning_module.trainer.datamodule

        fig, ax = plt.subplots()
        for i, dm_val in enumerate(dm_super.data_modules_val):
            c, steps = get_loss_steps(
                f"{dm_val.dataset_name}_val_crps{suffix}", trainer
            )
            ax.semilogy(
                steps,
                c,
                label=dm_val.dataset_name,
                alpha=1.0,
                color=cm.colors[i],
            )

        macro, steps = get_loss_steps(f"val_crps{suffix}_macro", trainer)
        ax.semilogy(steps, macro, "r--", label=f"val_crps{suffix}_macro")
        ticks = range(0, trainer.current_epoch + 1, self.every_n_epochs)
        labels = [str(t) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(
            ncol=2,
            loc="upper center",
            handletextpad=1.5,
            bbox_to_anchor=(0.5, 1.3),
            framealpha=0.5,
            prop={"size": 8},
        )

        fig.savefig(
            get_save_dir_from_csvlogger(trainer.logger)
            / f"val_crps{suffix}_macro.png",
            bbox_inches="tight",
        )
        plt.close(fig)
