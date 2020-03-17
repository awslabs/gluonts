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

# First-party imports
import random

# Third-party imports
import numpy as np
import pandas as pd
from mxnet.context import current_context

# First-party imports
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.dataset.common import ListDataset
from gluonts.transform import Chain, UniformSplitSampler, InstanceSplitter
from gluonts.dataset.artificial import ConstantDataset

# CONSTANTS:

BATCH_SIZE = 8
NUM_WORKERS_MP = 5
NUM_WORKERS = 0
CONTEXT_LEN = 14
SPLITTING_SAMPLE_PROBABILITY = 1  # crucial for the ValidationDataLoader test
CD_NUM_STEPS = 30
CD_NUM_TIME_SERIES = 100
CD_MAX_LEN_MULTIPLICATION_FACTOR = 5

# get dataset and deterministic transformation
def get_dataset_and_transformation():
    # create constant dataset with each time series having
    # variable length and unique constant integer entries
    dataset = ConstantDataset(
        num_steps=CD_NUM_STEPS, num_timeseries=CD_NUM_TIME_SERIES
    )
    list_dataset = list(dataset.train)
    for i, ts in enumerate(list_dataset):
        ts["start"] = pd.Timestamp(ts_input=ts["start"], freq=dataset.freq)
        # get randomness in the ts lengths
        ts["target"] = np.array(
            ts["target"] * random.randint(1, CD_MAX_LEN_MULTIPLICATION_FACTOR)
        )
    list_dataset = ListDataset(data_iter=list_dataset, freq=dataset.freq)
    list_dataset_pred_length = dataset.prediction_length

    # use every possible time point to split the time series
    transformation = Chain(
        [
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=UniformSplitSampler(
                    p=SPLITTING_SAMPLE_PROBABILITY  # THIS IS IMPORTANT FOR THE TEST
                ),
                past_length=CONTEXT_LEN,
                future_length=list_dataset_pred_length,
                dummy_value=1.0,
            ),
        ]
    )

    return list_dataset, transformation, list_dataset_pred_length


# returns the number of transformed datasets per original ts as a dict
def get_transformation_counts(dataset):
    transformation_counts = {}
    for batch in dataset:
        for ts_id in batch["item_id"]:
            ts_id = int(ts_id)
            if not ts_id in transformation_counts:
                transformation_counts[ts_id] = 1
            else:
                transformation_counts[ts_id] += 1
    return transformation_counts


# The idea is to test that the validation data loader yields equivalent results
def test_validation_loader_equivalence() -> None:
    (
        list_dataset,
        transformation,
        list_dataset_pred_length,
    ) = get_dataset_and_transformation()
    current_desired_context = current_context()

    validation_dataset_loader = ValidationDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
        ctx=current_desired_context,
    )

    # multi-processed validation dataset
    mp_val_data_loader_result_01 = list(validation_dataset_loader)

    # multi-processed validation dataset NR2, second iteration/pass through
    mp_val_data_loader_result_02 = list(validation_dataset_loader)

    # single-processed validation dataset
    training_data_loader_result = list(
        ValidationDataLoader(
            dataset=list_dataset,
            transform=transformation,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,  # This is the crucial difference
            ctx=current_desired_context,
        )
    )

    # ASSERTIONS:

    assert get_transformation_counts(
        mp_val_data_loader_result_01
    ) == get_transformation_counts(
        training_data_loader_result
    ), "The multiprocessing ValidationDataLoader should yield equivalent result to the non multiprocessing one."

    assert get_transformation_counts(
        mp_val_data_loader_result_02
    ) == get_transformation_counts(
        training_data_loader_result
    ), "The multiprocessing ValidationDataLoader should yield equivalent result to the non multiprocessing one."

    assert (
        len(mp_val_data_loader_result_02[0]["item_id"]) == BATCH_SIZE
    ), "Incorrect batch size from multiprocessing."

    assert (
        mp_val_data_loader_result_02[0]["past_target"].context
        == current_desired_context
    ), "Batches in incorrect context"


# The idea of the test is, that:
# CASE 01: if we have say 5 workers, then iterating
# over the dataset so that one worker could cover 2/5 of the whole dataset
# should still be enough that every time series is at least processed once
# CASE 02: if we have say 5 workers, but let each only cover 1/10, then
# it should be impossible to cover the whole underlying dataset
# CASE 03: one worker should be able to traverse the whole dataset on its own
def test_training_loader_soft_constraint() -> None:
    (
        list_dataset,
        transformation,
        list_dataset_pred_length,
    ) = get_dataset_and_transformation()
    current_desired_context = current_context()

    # the on average expected processed time series samples, and batches
    avg_mult_factor = (
        sum(range(1, CD_MAX_LEN_MULTIPLICATION_FACTOR + 1))
        / CD_MAX_LEN_MULTIPLICATION_FACTOR
    )
    exp_num_samples = CD_NUM_TIME_SERIES * (
        avg_mult_factor * CD_NUM_STEPS - CONTEXT_LEN - list_dataset_pred_length
    )
    exp_num_batches = int(exp_num_samples / BATCH_SIZE)

    # CASE 01: EVERY TS VISITED ONCE

    train_dataset_loader_01 = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
        ctx=current_desired_context,
        num_batches_per_epoch=int(2 * exp_num_batches),
    )

    # multi-processed validation dataset
    mp_training_data_loader_result_01 = list(train_dataset_loader_01)

    # should contain an entry for every time series id
    transformation_counts_01 = get_transformation_counts(
        mp_training_data_loader_result_01
    )

    assert all(
        [k in transformation_counts_01 for k in range(CD_NUM_TIME_SERIES)]
    ), "Not every time series processed at least once."

    # CASE 02: NOT EVERY TS VISITED ONCE

    train_dataset_loader_02 = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
        ctx=current_desired_context,
        num_batches_per_epoch=int(0.5 * exp_num_batches),
    )

    # multi-processed validation dataset
    mp_training_data_loader_result_02 = list(train_dataset_loader_02)

    # should contain an entry for every time series id
    transformation_counts_02 = get_transformation_counts(
        mp_training_data_loader_result_02
    )

    assert not all(
        [k in transformation_counts_02 for k in range(CD_NUM_TIME_SERIES)]
    ), "It should not have been possible to process every time series once. "

    # CASE 03: ONE WORKER TRAVERSES ALL

    train_dataset_loader_03 = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        num_workers=1,  # This is the crucial difference
        ctx=current_desired_context,
        num_batches_per_epoch=int(2 * exp_num_batches),
    )

    # multi-processed validation dataset
    mp_training_data_loader_result_03 = list(train_dataset_loader_03)

    # should contain an entry for every time series id
    transformation_counts_03 = get_transformation_counts(
        mp_training_data_loader_result_03
    )

    assert all(
        [k in transformation_counts_03 for k in range(CD_NUM_TIME_SERIES)]
    ), "One worker should be able to traverse all in one sweep, and should not deplete its iterator."
