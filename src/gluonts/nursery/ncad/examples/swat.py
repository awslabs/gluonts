# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
import time
from pathlib import Path, PosixPath

from typing import Optional, Union

import itertools

import numpy as np
import json

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import ncad
from ncad.ts import TimeSeriesDataset
from ncad.ts import transforms as tr
from ncad.model import NCAD, NCADDataModule


def swat_inject_anomalies(
    dataset: TimeSeriesDataset,
    injection_method: str = ["None", "local_outliers"][-1],
    ratio_injected_spikes: float = None,
) -> TimeSeriesDataset:

    # dataset is transformed using a TimeSeriesTransform depending on the type of injection

    if injection_method == "None":
        return dataset
    elif injection_method == "local_outliers":
        if ratio_injected_spikes is None:
            # Inject synthetic anomalies: LocalOutlier
            ts_transform = tr.LocalOutlier(area_radius=500, num_spikes=360)
        else:
            ts_transform = tr.LocalOutlier(
                area_radius=700,
                num_spikes=ratio_injected_spikes,
                spike_multiplier_range=(1.0, 3.0),
                # direction_options = ['increase'],
            )

        # Generate many examples of injected time series
        multiplier = 20
        ts_transform_iterator = ts_transform(itertools.cycle(dataset))
        dataset_transformed = ncad.utils.take_n_cycle(
            ts_transform_iterator, multiplier * len(dataset)
        )
        dataset_transformed = TimeSeriesDataset(dataset_transformed)
    else:
        raise ValueError(f"injection_method = {injection_method} not supported!")

    return dataset_transformed


def swat_pipeline(
    data_dir: Union[str, PosixPath],
    model_dir: Union[str, PosixPath],
    log_dir: Union[str, PosixPath],
    ## General
    exp_name: Optional[str] = None,
    ## For trainer
    epochs: int = 500,
    gpus: int = 1 if torch.cuda.is_available() else 0,
    limit_val_batches: float = 1.0,
    num_sanity_val_steps: int = 1,
    ## For injection
    injection_method: str = ["None", "local_outliers"][-1],
    ratio_injected_spikes: float = None,
    ## For DataLoader
    window_length: int = 500,
    suspect_window_length: int = 10,
    num_series_in_train_batch: int = 1,
    num_crops_per_series: int = 32,
    num_workers_loader: int = 0,
    ## For model definition
    # hpars for encoder
    tcn_kernel_size: int = 7,
    tcn_layers: int = 10,
    tcn_out_channels: int = 16,
    tcn_maxpool_out_channels: int = 32,
    embedding_rep_dim: int = 64,
    normalize_embedding: bool = True,
    # hpars for classifier
    distance: str = ["cosine", "L2", "non-contrastive"][0],
    classifier_threshold: float = 0.5,
    threshold_grid_length_val: float = 0.10,
    threshold_grid_length_test: float = 0.05,
    # hpars for anomalizers
    coe_rate: float = 0.5,
    mixup_rate: float = 2.0,
    # hpars for optimizer
    learning_rate: float = 1e-4,
    # hpars for validation and test
    check_val_every_n_epoch: int = 25,
    stride_roll_pred_val_test: int = 5,
    val_labels_adj: bool = True,
    test_labels_adj: bool = True,
    max_windows_unfold_batch: Optional[int] = 5000,
    evaluation_result_path: Optional[Union[str, PosixPath]] = None,
    # For reproducibility
    rnd_seed: int = 123,
    **kwargs,
):

    # Expand user path
    dirs = [data_dir, model_dir, log_dir]
    data_dir, model_dir, log_dir = [
        PosixPath(path).expanduser() if str(path).startswith("~") else Path(path) for path in dirs
    ]

    # Create directories if inexistent
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if (not os.path.exists(log_dir)) and (not str(log_dir).startswith("s3://")):
        os.makedirs(log_dir)

    # Set random seed #
    pl.trainer.seed_everything(rnd_seed)

    #####     Load Data     #####

    train_set, test_set = ncad.datasets.swat(
        path=data_dir,
        subsample_one_in_k=10,
        subsample_fun=["mean", "median"][0],
        multivariate=True,
    )
    # Standardize TimeSeries values (substract median, divide by interquartile range)
    scaler = tr.TimeSeriesScaler(type="robust")
    train_set = TimeSeriesDataset(ncad.utils.take_n_cycle(scaler(train_set), len(train_set)))
    test_set = TimeSeriesDataset(ncad.utils.take_n_cycle(scaler(test_set), len(test_set)))
    # Number of channels in TimeSeries
    ts_channels = train_set[0].shape[1]
    assert all(shape[1] == ts_channels for shape in train_set.shape)
    assert all(shape[1] == ts_channels for shape in test_set.shape)

    #### inject anomalies on train dataset ###
    train_set_transformed = swat_inject_anomalies(
        dataset=train_set,
        injection_method=injection_method,
        ratio_injected_spikes=ratio_injected_spikes,
    )

    # Split test dataset in two, half for validation and half for test
    _, validation_set, test_set = ncad.ts.split_train_val_test(
        data=test_set,
        val_portion=0.5,  # No labels in training data that could serve for validation
        test_portion=0.5,  # This dataset provides test set, so we don't need to take from training data
        split_method="past_future_with_warmup",
        split_warmup_length=window_length - suspect_window_length,
        verbose=False,
    )

    # Define DataModule for training with lighting (window cropping + window injection)#
    data_module = NCADDataModule(
        train_ts_dataset=train_set_transformed,
        validation_ts_dataset=validation_set,
        test_ts_dataset=test_set,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        num_series_in_train_batch=num_series_in_train_batch,
        num_crops_per_series=num_crops_per_series,
        label_reduction_method="any",
        stride_val_and_test=stride_roll_pred_val_test,
        num_workers=num_workers_loader,
    )

    if distance == "cosine":
        # For the contrastive approach, the cosine distance is used
        distance = ncad.model.distances.CosineDistance()
    elif distance == "L2":
        # For the contrastive approach, the L2 distance is used
        distance = ncad.model.distances.LpDistance(p=2)
    elif distance == "non-contrastive":
        # For the non-contrastive approach, the classifier is
        # a neural-net based on the embedding of the whole window
        distance = ncad.model.distances.BinaryOnX1(rep_dim=embedding_rep_dim, layers=1)

    # Instantiate model #
    model = NCAD(
        ts_channels=ts_channels,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        # hpars for encoder
        tcn_kernel_size=tcn_kernel_size,
        tcn_layers=tcn_layers,
        tcn_out_channels=tcn_out_channels,
        tcn_maxpool_out_channels=tcn_maxpool_out_channels,
        embedding_rep_dim=embedding_rep_dim,
        normalize_embedding=normalize_embedding,
        # hpars for classifier
        distance=distance,
        classification_loss=nn.BCELoss(),
        classifier_threshold=classifier_threshold,
        threshold_grid_length_test=threshold_grid_length_test,
        # hpars for anomalizers
        coe_rate=coe_rate,
        mixup_rate=mixup_rate,
        # hpars for validation and test
        stride_rolling_val_test=stride_roll_pred_val_test,
        val_labels_adj=val_labels_adj,
        test_labels_adj=test_labels_adj,
        max_windows_unfold_batch=max_windows_unfold_batch,
        # hpars for optimizer
        learning_rate=learning_rate,
    )

    # Experiment name #
    if exp_name is None:
        time_now = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
        exp_name = f"swat-{time_now}"

    ### Training the model ###

    logger = TensorBoardLogger(save_dir=log_dir, name=exp_name)

    # Checkpoint callback, monitoring 'val_f1'
    checkpoint_cb = ModelCheckpoint(
        monitor="val_f1",
        dirpath=model_dir,
        filename="ncad-model-" + exp_name + "-{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,
        mode="max",
    )

    # Checkpoint callback, monitoring 'train_loss_step'
    # checkpoint_cb = ModelCheckpoint(
    #     monitor='train_loss_step',
    #     dirpath=model_dir,
    #     filename='ncad-model-'+exp_name+'-{epoch:02d}-{train_loss_step:.4f}',
    #     save_top_k=1,
    #     mode='min',
    # )

    # Set training type in model and data module
    trainer = Trainer(
        gpus=gpus,
        default_root_dir=model_dir,
        logger=logger,
        min_epochs=epochs,
        max_epochs=epochs,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_cb],
        # callbacks=[checkpoint_cb, earlystop_cb, lr_logger],
        auto_lr_find=False,
    )

    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(
    #     model=model,
    #     datamodule=data_module,
    #     early_stop_threshold=None,
    #     num_training=50,
    # )
    # lr_finder.plot()
    # model.learning_rate = lr_finder.suggestion()
    # model.hparams.learning_rate = model.learning_rate

    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    # Load top performing checkpoint
    # ckpt_path = [x for x in model_dir.glob('*.ckpt')][-1]
    ckpt_file = [
        file
        for file in os.listdir(model_dir)
        if (file.endswith(".ckpt") and file.startswith("ncad-model-" + exp_name))
    ][-1]
    ckpt_path = model_dir / ckpt_file
    model = NCAD.load_from_checkpoint(ckpt_path)

    # Metrics on validation and test data #
    evaluation_result = trainer.test()
    evaluation_result = evaluation_result[0]

    # Save evaluation results
    if evaluation_result_path is not None:
        path = evaluation_result_path
        path = PosixPath(path).expanduser() if str(path).startswith("~") else Path(path)
        with open(path, "w") as f:
            json.dump(evaluation_result, f, cls=ncad.utils.NpEncoder)

    for key, value in evaluation_result.items():
        # if key.startswith('test_'):
        print(f"{key}={value}")

    print(f"NCAD on SWaT dataset finished successfully!")


from general_parser import get_general_parser
from ncad.utils import save_args

if __name__ == "__main__":

    # General parser:
    parser = get_general_parser()

    # SWaT specific parsing:

    args, _ = parser.parse_known_args()

    args_dict = vars(args)  # arguments as dictionary

    # Save parsed arguments
    model_dir = args_dict["model_dir"].expanduser()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_args(args=args_dict, path=model_dir / "args.json")

    swat_pipeline(**args_dict)
