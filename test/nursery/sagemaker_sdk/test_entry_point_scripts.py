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


# Standard library imports
import tempfile
from pathlib import Path
import argparse

# Third-party imports
import pytest

# First-party imports
from gluonts.core import serde
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.nursery.sagemaker_sdk.entry_point_scripts.train_entry_point import (
    train,
)
from gluonts.nursery.sagemaker_sdk.defaults import QUANTILES, NUM_SAMPLES


def create_arguments(temp_dir_abs_path, dataset_name, s3_dataset_path=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=temp_dir_abs_path)
    parser.add_argument(
        "--output-data_dir", type=str, default=temp_dir_abs_path
    )
    parser.add_argument("--estimator", type=str, default=temp_dir_abs_path)
    parser.add_argument("--s3-dataset", type=str, default=s3_dataset_path)
    parser.add_argument("--dataset", type=str, default=dataset_name)
    parser.add_argument("--num-samples", type=int, default=str(NUM_SAMPLES))
    parser.add_argument("--quantiles", type=str, default=str(QUANTILES))

    args, _ = parser.parse_known_args()

    return args


def simple_feedforward_estimator():
    return (
        SimpleFeedForwardEstimator,
        dict(
            ctx="cpu",
            epochs=1,
            learning_rate=1e-2,
            hybridize=True,
            num_hidden_dimensions=[3],
            num_batches_per_epoch=1,
            use_symbol_block_predictor=True,
            num_parallel_samples=1,
        ),
    )


@pytest.mark.parametrize(
    "dataset_name, custom_dataset", [("m4_hourly", False), ("m4_hourly", True)]
)
def test_train_script(dataset_name, custom_dataset):
    # we need to write some data for this test, so we use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        dataset = get_dataset(
            dataset_name, path=temp_dir_path, regenerate=True
        )  # exchange_rate, m4_yearly

        # either use provided dataset, in which case it must be present in the directory, or a built in one
        # for testing we will provide a built in dataset as a custom one too
        if custom_dataset:
            args = create_arguments(
                str(temp_dir_path),
                dataset_name,
                s3_dataset_path=str(temp_dir_path / dataset_name),
            )
        else:
            args = create_arguments(str(temp_dir_path), dataset_name)

        # the test requires using a deserialized estimator, which we first need to create
        estimator_cls, hyperparameters = simple_feedforward_estimator()
        estimator = estimator_cls.from_hyperparameters(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            **hyperparameters
        )
        serialized = serde.dump_json(estimator)
        with open(temp_dir_path / "estimator.json", "w") as estimator_file:
            estimator_file.write(serialized)

        # No assert necessary, the idea is just that the code needs to run through
        train(args)
