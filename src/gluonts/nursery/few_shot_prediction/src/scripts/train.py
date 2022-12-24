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

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import cast, Optional
from pathlib import Path
import os
import random
import click
import pytorch_lightning as pl
import pytorch_lightning.loggers as pll
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from meta.models.module import MetaLightningModule
from meta.models.common import load_weights
from meta.models import get_model, MODEL_REGISTRY
from meta.datasets import get_data_module, DATA_MODULE_REGISTRY
from meta.metrics import CRPS, QuantileLoss, QuantileWidth
from meta.callbacks.common import get_save_dir_from_csvlogger
from meta.callbacks import (
    ForecastPlotLoggerCallback,
    ParameterCountCallback,
    InitialSaveCallback,
    ForecastSupportSetAttentionPlotLoggerCallback,
    LossPlotLoggerCallback,
    CheatLossPlotLoggerCallback,
    MacroCRPSPlotCallback,
)
from experiments.evaluation.eval_real_cheat import large_training_evaluation


@click.command()
@click.option(
    "--data_dir",
    required=True,
    default=(
        cast(str, os.getenv("SM_CHANNEL_DATASETS"))
        if os.getenv("SM_INPUT_DIR") is not None
        else Path.home() / ".mxnet" / "gluon-ts"
    ),
)
@click.option(
    "--output_dir",
    required=True,
    default=os.getenv("SM_MODEL_DIR"),
)
@click.option("--seed", default=0)
# Experiment setup
@click.option(
    "--model_name",
    required=True,
    help="The model to fit on the data and to use for predictions.",
    type=click.Choice(MODEL_REGISTRY.keys()),
)
@click.option("--quantiles", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
@click.option(
    "--dm_name", required=True, type=click.Choice(DATA_MODULE_REGISTRY.keys())
)
@click.option("--support_set_size", default=3)
@click.option("--standardize", default=True)
# ------  Data module specific hyperparameters ------
# super data module
@click.option("--dm_super_dataset_names_train", default="electricity")
@click.option("--dm_super_dataset_names_val", default="electricity")
@click.option("--dm_super_dataset_names_test", default="electricity")
@click.option(
    "--dm_super_dataset_sampling",
    default="weighted",
    type=click.Choice(["weighted", "uniform"]),
)
@click.option("--dm_super_catch22_train", default=False)
@click.option("--dm_super_catch22_val_test", default=False)
@click.option("--dm_super_cheat", default=0.0)
@click.option("--dm_super_num_queries", default=1)
# artificial data module
@click.option("--dm_artificial_dataset_name", default="marker")
@click.option("--dm_artificial_prediction_length", default=24)
@click.option("--dm_artificial_train_set_size", default=10000)
# artifical cheat data module
@click.option("--dm_cheat_train_set_size", default=10000)
@click.option("--dm_cheat_val_set_size", default=500)
@click.option("--dm_cheat_test_set_size", default=500)
@click.option("--dm_cheat_prediction_length", default=24)
@click.option("--dm_cheat_cheat_chance", default=1.0)
@click.option("--dm_cheat_noise_level", default=0.0)
@click.option("--dm_cheat_query_length_scale", default=5.0)
@click.option("--dm_cheat_support_length_scale_same_as_query", default=True)
# artifical cheat counterfactual data module
@click.option("--dm_cheat_counterfactual_train_set_size", default=10000)
@click.option("--dm_cheat_counterfactual_val_set_size", default=500)
@click.option("--dm_cheat_counterfactual_test_set_size", default=500)
@click.option("--dm_cheat_counterfactual_prediction_length", default=24)
@click.option("--dm_cheat_counterfactual_cheat_chance", default=1.0)
@click.option("--dm_cheat_counterfactual_noise_level", default=0.0)
@click.option("--dm_cheat_counterfactual_query_length_scale", default=5.0)
@click.option("--dm_cheat_counterfactual_counterfactual_size", default=1)
@click.option("--dm_cheat_counterfactual_counterfactual_mixing", default=False)
@click.option(
    "--dm_cheat_counterfactual_support_length_scale_same_as_query",
    default=True,
)
# Common hyperparameters
@click.option("--pretrained", default=None)
@click.option("--learning_rate", default=1e-3)
@click.option("--lr_on_plateau_patience", default=5)
@click.option("--lr_on_plateau_factor", default=0.3)
@click.option("--context_length_multiple", default=4)
@click.option("--support_length_multiple", default=4)
@click.option("--model_prediction_length", default=0)
@click.option("--max_epochs", default=400)
@click.option("--max_training_time", default="02:00:00:00")
@click.option("--num_batches", default=1024)
@click.option("--batch_size_train", default=1000)
@click.option("--batch_size_val_test", default=1000)
@click.option("--accumulate_grad_batches", default=0)
@click.option("--early_stopping_min_delta", default=0.001)
@click.option("--early_stopping_patience", default=20)
@click.option(
    "--monitor_metric",
    required=True,
    help="This metric is used for EarlyStopping, ModelCheckpoint, LRScheduler callbacks.",
)
@click.option("--n_logging_samples", default=5)
@click.option("--log_plot_every_n_epochs", default=10)
@click.option("--num_workers", default=8)
# ------  Model specific hyperparameters ------
# lstm feedforward model
@click.option("--lstm_feedforward_num_lstm_layers", default=1)
@click.option("--lstm_feedforward_query_out_channels", default=32)
@click.option("--lstm_feedforward_decoder_hidden_size", default=32)
# CNN Iwata model
@click.option("--cnn_iwata_num_heads", default=1)
@click.option("--cnn_iwata_supps_kernel_size", default=3)
@click.option("--cnn_iwata_supps_out_channels", default=32)
# Iwata model
@click.option("--iwata_num_heads", default=1)
@click.option("--iwata_supps_out_channels", default=32)
@click.option("--iwata_query_out_channels", default=32)
@click.option("--iwata_supps_bidirectional", default=True)
@click.option("--iwata_decoder_hidden_size", default=32)
@click.option("--iwata_lstm_num_layers", default=1)
# TCN model
@click.option("--tcn_num_heads", default=1)
@click.option("--tcn_num_channels", default=64)
@click.option("--tcn_kernel_size", default=2)
@click.option("--tcn_num_layers", default=5)
@click.option("--tcn_hidden_size", default=64)
@click.option("--tcn_decoder_hidden_size", default=64)
def main(
    data_dir: str,
    output_dir: str,
    seed: int,
    # Experiment setup
    model_name: str,
    quantiles: str,
    dm_name: str,
    support_set_size: int,
    standardize: bool,
    # Common hyperparameters
    pretrained: str,
    learning_rate: float,
    lr_on_plateau_patience: float,
    lr_on_plateau_factor: float,
    context_length_multiple: int,
    support_length_multiple: int,
    model_prediction_length: Optional[int],
    max_epochs: int,
    max_training_time: str,
    num_batches: int,
    batch_size_train: int,
    batch_size_val_test: int,
    accumulate_grad_batches: Optional[int],
    early_stopping_min_delta: float,
    early_stopping_patience: int,
    monitor_metric: str,
    n_logging_samples: int,
    log_plot_every_n_epochs: int,
    num_workers: int,
    # Model hyperparameters
    **kwargs: int,
):
    """
    Trains a model in the meta learning framework
    """
    args_to_save = locals()
    random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    kwargs["dm_super_seed"] = seed

    quantiles = quantiles.split(",")

    # -------------------- Configure the data loader ---------------------------------------------------
    dm = get_data_module(
        dm_name,
        support_set_size=support_set_size,
        support_length_multiple=support_length_multiple,
        standardize=standardize,
        data_dir=Path(data_dir),
        batch_size_train=batch_size_train,
        batch_size_val_test=batch_size_val_test,
        context_length_multiple=context_length_multiple,
        num_workers=num_workers,
        **{
            key[len(dm_name) + 1 :]: value
            for key, value in kwargs.items()
            if key.startswith(dm_name)
        },
    )
    dm.setup()
    # -------------------- Initialize the model ---------------------------------------------------
    model = get_model(
        model_name,
        prediction_length=model_prediction_length or dm.prediction_length,
        quantiles=quantiles,
        **{
            key[len(model_name) + 1 :]: value
            for key, value in kwargs.items()
            if key.startswith(model_name)
        },
    )
    if pretrained:
        model = load_weights(model=model, path_to_weights=pretrained)

    module = MetaLightningModule(
        model,
        loss=QuantileLoss(quantiles),
        crps=CRPS(quantiles, rescale=dm.standardize),
        lr_scheduler_monitor=monitor_metric,
        crps_scaled=CRPS(quantiles) if dm.standardize else None,
        quantile_width=QuantileWidth(quantiles),
        lr=learning_rate,
        lr_on_plateau_patience=lr_on_plateau_patience,
        lr_on_plateau_factor=lr_on_plateau_factor,
        val_dataset_names=dm.dataset_names_val,
        test_dataset_names=dm.dataset_names_test,
    )

    # -------------------- configure callbacks ---------------------------------------------------
    # for the plotting callback we need to get some validation data
    log_batch_train, log_batch_val = dm.get_log_batches(n_logging_samples)

    # early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        filename=monitor_metric + "--{epoch:02d}--{train_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    callbacks = [
        ParameterCountCallback(),
        InitialSaveCallback(args_to_save=args_to_save),
        early_stop_callback,
        checkpoint_callback,
    ]
    if dm_name == "dm_super":
        callbacks.append(
            LossPlotLoggerCallback(every_n_epochs=log_plot_every_n_epochs)
        )
        if len(kwargs["dm_super_dataset_names_val"].split(",")) > 1:
            callbacks.append(
                MacroCRPSPlotCallback(every_n_epochs=log_plot_every_n_epochs)
            )
    elif dm_name == "dm_cheat":
        callbacks.append(
            CheatLossPlotLoggerCallback(
                dataset_names_val=dm.dataset_names_val,
                every_n_epochs=log_plot_every_n_epochs,
            )
        )

    # add callback that only works with attention
    attention_models = ["iwata", "cnn_iwata", "tcn"]
    if model_name in attention_models:
        callbacks.extend(
            [
                ForecastSupportSetAttentionPlotLoggerCallback(
                    log_batch_train,
                    quantiles=quantiles,
                    split="train",
                    every_n_epochs=log_plot_every_n_epochs,
                ),
                ForecastSupportSetAttentionPlotLoggerCallback(
                    log_batch_val,
                    quantiles=quantiles,
                    split="val",
                    every_n_epochs=log_plot_every_n_epochs,
                ),
            ]
        )
    else:
        callbacks.extend(
            [
                ForecastPlotLoggerCallback(
                    log_batch_val,
                    quantiles=quantiles,
                    split="val",
                    every_n_epochs=log_plot_every_n_epochs,
                ),
                ForecastPlotLoggerCallback(
                    log_batch_train,
                    quantiles=quantiles,
                    split="train",
                    every_n_epochs=log_plot_every_n_epochs,
                ),
            ]
        )

    # -------------------- train model ---------------------------------------------------
    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        limit_train_batches=num_batches,
        val_check_interval=1.0,
        max_epochs=max_epochs,
        max_time=max_training_time,
        logger=pll.CSVLogger(output_dir),
        accumulate_grad_batches=accumulate_grad_batches or None,
        callbacks=callbacks,
        # profiler="pytorch",
    )

    trainer.fit(module, dm)
    trainer.test(module, dm, ckpt_path="best")

    # -------------------- evaluate model and store results ---------------------------------------------------
    # load weights with best validation loss during training period
    model_best = load_weights(
        model=model, path_to_weights=checkpoint_callback.best_model_path
    )
    if dm_name == "dm_super" and model_name in attention_models:
        save_dir = get_save_dir_from_csvlogger(trainer.logger)
        large_training_evaluation(
            save_dir=save_dir,
            model_name=model_name,
            seed=seed,
            support_length_multiple=support_length_multiple,
            dm_super_cheat=kwargs["dm_super_cheat"],
            datasets=kwargs["dm_super_dataset_names_test"].split(","),
            dm_super_catch22_train=kwargs["dm_super_catch22_train"],
            quantiles=quantiles,
            model=model_best,
            data_dir=Path(data_dir),
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
