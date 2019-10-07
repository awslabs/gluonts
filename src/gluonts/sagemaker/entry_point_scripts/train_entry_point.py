# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

# Standard library imports
from pathlib import Path
import argparse
import os
import json
import time

# Third-party imports


# First-party imports
from gluonts.core import serde
from gluonts.dataset import common
from gluonts.dataset.repository import datasets
from gluonts import sagemaker
from gluonts.evaluation import backtest


def train(arguments):
    # Generic gluonts training method

    # TODO fix paths

    print(arguments)

    # load the dataset
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), " Downloading - Downloading dataset.")

    # deserialize the estimator
    estimator_config = Path(arguments.estimator) / "estimator.json"
    estimator = None
    with open(estimator_config, 'r') as f:
        estimator = serde.load_json(f.read(f))

    # load the dataset into gluonts format
    dataset = None
    if arguments.SM_CHANNEL_S3_DATASET == "None":
        # load built in dataset
        dataset = datasets.get_dataset(arguments.sm_hps[sagemaker.GluonTSFramework.DATASET])
    else:
        # load custom dataset
        s3_dataset_dir = Path(arguments.s3_dataset)
        dataset = common.load_datasets(metadata=s3_dataset_dir,
                                       train=s3_dataset_dir / "train",
                                       test=s3_dataset_dir / "test")

    # train and evaluate the models
    aggregate_metrics, per_time_series_metrics = backtest.backtest_metrics(
        train_dataset=dataset.train,
        test_dataset=dataset.test,
        forecaster=estimator,
    )

    # save the evaluation results to the right location

    # save the model to the right location


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument('--sm_hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])

    parser.add_argument('--estimator', type=str, default=os.environ['SM_CHANNEL_ESTIMATOR'])

    # argument possibly not set
    parser.add_argument('--s3_dataset}', type=str, default=str(os.environ.get('SM_CHANNEL_S3_DATASET')))

    args, _ = parser.parse_known_args()

    train(args)
