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
from pathlib import Path, PosixPath
import argparse

import torch

from pytorch_lightning.utilities.parsing import str_to_bool

from ncad.utils import save_args


def get_general_parser():
    parser = argparse.ArgumentParser()

    if "SM_MODEL_DIR" in os.environ:
        parser.add_argument("--data_dir", type=PosixPath, default=os.environ.get("SM_CHANNEL_DATA"))
        parser.add_argument(
            "--model_dir", type=PosixPath, default=os.environ["SM_MODEL_DIR"]
        )  # To store the trained model
        parser.add_argument(
            "--log_dir", type=PosixPath, default=os.environ["SM_OUTPUT_DATA_DIR"]
        )  # To store tensorboard log
    else:
        parser.add_argument("--data_dir", type=PosixPath)
        parser.add_argument("--model_dir", type=PosixPath)  # To store the trained model
        parser.add_argument("--log_dir", type=PosixPath)  # To store tensorboard logs

    # General:
    parser.add_argument("--exp_name", type=str, default=None)

    ## For trainer
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--gpus", type=int, default=1 if torch.cuda.is_available() else 0)
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="How much of validation dataset to check. Should be between 0 and 1.",
    )

    ## For injection methods
    parser.add_argument(
        "--injection_method", type=str, choices=["None", "local_outliers"], default="local_outliers"
    )
    parser.add_argument("--ratio_injected_spikes", type=float, default=None)

    ## For DataLoader
    parser.add_argument("--window_length", type=int, default=150)
    parser.add_argument("--suspect_window_length", type=int, default=5)

    parser.add_argument("--validation_portion", type=float, default=0.2)
    parser.add_argument("--test_portion", type=float, default=0.3)
    parser.add_argument(
        "--train_split_method",
        type=str,
        choices=["random_series", "sequential", "past_future", "past_future_with_warmup"],
        default="past_future_with_warmup",
    )

    parser.add_argument("--num_series_in_batch", type=int, default=16)
    parser.add_argument("--num_crops_per_series", type=int, default=16)

    parser.add_argument(
        "--rate_true_anomalies",
        type=float,
        default=0.0,
        help="Percentage of true anomalies that are labelled in the training data",
    )
    parser.add_argument("--num_workers_loader", type=int, default=0)

    ## For model definition
    # hpars for encoder
    parser.add_argument("--tcn_kernel_size", type=int, default=7)
    parser.add_argument("--tcn_layers", type=int, default=3)
    parser.add_argument("--tcn_out_channels", type=int, default=20)
    parser.add_argument("--tcn_maxpool_out_channels", type=int, default=16)
    parser.add_argument("--embedding_rep_dim", type=int, default=150)
    parser.add_argument("--normalize_embedding", type=str_to_bool, default=True)
    # hpars for classifier
    parser.add_argument("--distance", type=str, default="L2")
    parser.add_argument("--classifier_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_grid_length_val", type=float, default=0.10)
    parser.add_argument("--threshold_grid_length_test", type=float, default=0.05)
    # hpars for anomalizers
    parser.add_argument("--coe_rate", type=float, default=0.5)
    parser.add_argument("--mixup_rate", type=float, default=2.0)
    # hpars for optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    # hpars for validation and test
    parser.add_argument("--check_val_every_n_epoch", type=int, default=25)
    parser.add_argument("--stride_roll_pred_val_test", type=int, default=1)
    parser.add_argument("--val_labels_adj", type=str_to_bool, default=True)
    parser.add_argument("--test_labels_adj", type=str_to_bool, default=True)
    parser.add_argument("--max_windows_unfold_batch", type=int, default=5000)
    parser.add_argument("--evaluation_result_path", type=PosixPath, default=None)
    # hpars for reproducibility
    parser.add_argument("--rnd_seed", type=int, default=123)
    # Dummy arg
    parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")

    return parser
