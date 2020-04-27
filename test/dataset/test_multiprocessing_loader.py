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
import json
import random
import tempfile
import time
import multiprocessing as mp

# Third-party imports
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from mxnet.context import current_context
from flaky import flaky
import pytest

# First-party imports
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    TrainDataLoader,
    ValidationDataLoader,
    InferenceDataLoader,
)
from gluonts.dataset.common import ListDataset, FileDataset
from gluonts.transform import (
    Chain,
    UniformSplitSampler,
    InstanceSplitter,
    InstanceSampler,
)
from gluonts.dataset.artificial import ConstantDataset

from gluonts.model.deepar import DeepAREstimator
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.trainer import Trainer
from gluonts.dataset.artificial import constant_dataset
from gluonts.evaluation import Evaluator

# CONSTANTS:

BATCH_SIZE = 8
NUM_WORKERS_MP = (
    5  # 5 is specific and intentional, see train set soft constraint test
)
CONTEXT_LEN = 7
SPLITTING_SAMPLE_PROBABILITY = 1  # crucial for the ValidationDataLoader test
CD_NUM_STEPS = 14
CD_NUM_TIME_SERIES = 47  # too small and batch test might fail
CD_MAX_LEN_MULTIPLICATION_FACTOR = 3

# NEEDED FOR SEGMENTATION COVERAGE TEST:
assert CD_NUM_TIME_SERIES % NUM_WORKERS_MP != 0

# CACHED DATA

_data_cache = None

# delete cache explicitly in last test
def delete_cache() -> None:
    global _data_cache
    if _data_cache is not None:
        del _data_cache


# get dataset and deterministic transformation
def get_dataset_and_transformation():
    # dont recompute, since expensive
    global _data_cache
    if _data_cache is not None:
        return _data_cache

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

    # original no multiprocessing processed validation dataset
    train_data_transformed_original = list(
        ValidationDataLoader(
            dataset=list_dataset,
            transform=transformation,
            batch_size=BATCH_SIZE,
            num_workers=0,  # This is the crucial difference
            ctx=current_context(),
        )
    )

    _data_cache = (
        list_dataset,
        transformation,
        list_dataset_pred_length,
        train_data_transformed_original,
    )

    return _data_cache


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
        train_data_transformed_original,
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

    # ASSERTIONS:

    assert len(list_dataset.list_data) == len(
        get_transformation_counts(mp_val_data_loader_result_01)
    ), "The dataloaders do not cover the whole dataset. Check that each time series was assigned at least one worker."

    assert get_transformation_counts(
        mp_val_data_loader_result_01
    ) == get_transformation_counts(
        train_data_transformed_original
    ), "The multiprocessing ValidationDataLoader should yield equivalent result to the non multiprocessing one."

    assert get_transformation_counts(
        mp_val_data_loader_result_02
    ) == get_transformation_counts(
        train_data_transformed_original
    ), "The multiprocessing ValidationDataLoader should yield equivalent result to the non multiprocessing one."

    assert (
        len(mp_val_data_loader_result_02[0]["item_id"]) == BATCH_SIZE
    ), "Incorrect batch size from multiprocessing."

    assert (
        mp_val_data_loader_result_02[0]["past_target"].context
        == current_desired_context
    ), "Batches in incorrect context"


@flaky(max_runs=5, min_passes=1)
@pytest.mark.parametrize(
    "num_workers",
    [i for i in [None, 1, 2,] if i is None or i <= mp.cpu_count()],
    # TODO: using more than 2 is a problem for our tests, if some of the cores are busy and fall behind
    # TODO: using multiple input queues in the loader would make this pass no matter how busy each core is
    # [i for i in [None, 1, 2, 3, 4] if i is None or i <= mp.cpu_count()],
)
def test_train_loader_goes_over_all_data(num_workers) -> None:
    batch_size = 4
    num_batches_per_epoch = 4

    X = 3

    simple_data = [
        {
            "start": "2012-01-01",
            "target": np.random.uniform(size=40).astype(float).tolist(),
            "item_id": i,
        }
        for i in range(batch_size * num_batches_per_epoch * X)
    ]

    num_passes = 5
    num_epochs = X * num_passes

    def test_dataset(dataset):
        class ExactlyOneSampler(InstanceSampler):
            def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
                window_size = b - a + 1
                assert window_size > 0
                return np.array([a])

        transformation = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            train_sampler=ExactlyOneSampler(),
            past_length=10,
            future_length=5,
            dummy_value=1.0,
        )

        dl = TrainDataLoader(
            dataset=dataset,
            transform=transformation,
            batch_size=batch_size,
            num_workers=num_workers,
            num_batches_per_epoch=num_batches_per_epoch,
            ctx=current_context(),
        )

        item_ids = defaultdict(int)

        for epoch in range(num_epochs):
            for batch in dl:
                for item_id in batch["item_id"]:
                    item_ids[item_id] += 1

        for i in range(len(dataset)):
            assert num_passes - 1 <= item_ids[i] <= num_passes + 1

    test_dataset(ListDataset(simple_data, freq="1H"))

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/data.json", "w") as f:
            for data in simple_data:
                json.dump(data, f)
                f.write("\n")

        test_dataset(FileDataset(Path(tmpdir), freq="1H"))
        test_dataset(FileDataset(Path(tmpdir), freq="1H", cache=True))


# The idea is to test that the inference data loader yields equivalent results
def test_inference_loader_equivalence() -> None:
    (
        list_dataset,
        transformation,
        list_dataset_pred_length,
        train_data_transformed_original,
    ) = get_dataset_and_transformation()
    current_desired_context = current_context()

    # original no multiprocessing processed validation dataset
    inference_loader_data_transformed_original = list(
        InferenceDataLoader(
            dataset=list_dataset,
            transform=transformation,
            batch_size=BATCH_SIZE,
            num_workers=0,  # This is the crucial difference
            ctx=current_context(),
        )
    )

    inference_loader = InferenceDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
        ctx=current_context(),
    )

    # multi-processed validation dataset
    mp_inf_data_loader_result_01 = list(inference_loader)

    # multi-processed validation dataset NR2, second iteration/pass through
    mp_inf_data_loader_result_02 = list(inference_loader)

    # ASSERTIONS:

    assert get_transformation_counts(
        mp_inf_data_loader_result_01
    ) == get_transformation_counts(
        inference_loader_data_transformed_original
    ), "The multiprocessing ValidationDataLoader should yield equivalent result to the non multiprocessing one."

    assert get_transformation_counts(
        mp_inf_data_loader_result_02
    ) == get_transformation_counts(
        inference_loader_data_transformed_original
    ), "The multiprocessing ValidationDataLoader should yield equivalent result to the non multiprocessing one."

    assert (
        len(mp_inf_data_loader_result_02[1]["item_id"]) == BATCH_SIZE
    ), "Incorrect batch size from multiprocessing."

    assert (
        mp_inf_data_loader_result_02[0]["past_target"].context
        == current_desired_context
    ), "Batches in incorrect context"


# CASE 01: if we have say 5 workers, then iterating
# over the dataset so that one worker could cover 3/5 of the whole dataset
# should still be enough that every time series is at least processed once,
@pytest.mark.xfail(
    reason="""High data-subset length variability, cheap batch transformations and
    different process start times often lead to this toy test failing."""
)
@flaky(max_runs=3, min_passes=1)
def test_training_loader_soft_constraint_01() -> None:
    (
        list_dataset,
        transformation,
        list_dataset_pred_length,
        train_data_transformed_original,
    ) = get_dataset_and_transformation()

    # the expected number of batches
    exp_num_batches = len(train_data_transformed_original)

    # CASE 01: EVERY TS VISITED AT LEAST ONCE

    train_dataset_loader_01 = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
        ctx=current_context(),
        num_batches_per_epoch=int(3 * exp_num_batches),
    )

    # give all the workers a little time to get ready, so they can start at the same time
    time.sleep(1.5)

    # multi-processed validation dataset
    mp_training_data_loader_result_01 = list(train_dataset_loader_01)

    # should contain an entry for every time series id
    transformation_counts_01 = get_transformation_counts(
        mp_training_data_loader_result_01
    )

    assert all(
        [k in transformation_counts_01 for k in range(CD_NUM_TIME_SERIES)]
    ), "Not every time series processed at least once."


# CASE 02: if we have say 5 workers, but let each only cover 1/10, then
# it should be impossible to cover the whole underlying dataset
@flaky(max_runs=2, min_passes=2)
def test_training_loader_soft_constraint_02() -> None:
    (
        list_dataset,
        transformation,
        list_dataset_pred_length,
        train_data_transformed_original,
    ) = get_dataset_and_transformation()

    # the expected number of batches
    exp_num_batches = len(train_data_transformed_original)

    # CASE 02: NOT EVERY TS VISITED ONCE

    train_dataset_loader_02 = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
        ctx=current_context(),
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


# CASE 03: one worker should be able to traverse the whole dataset on its own
def test_training_loader_soft_constraint_03() -> None:
    (
        list_dataset,
        transformation,
        list_dataset_pred_length,
        train_data_transformed_original,
    ) = get_dataset_and_transformation()

    # the expected number of batches
    exp_num_batches = len(train_data_transformed_original)

    # CASE 03: ONE WORKER TRAVERSES ALL

    train_dataset_loader_03 = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        num_workers=1,  # This is the crucial difference
        ctx=current_context(),
        num_batches_per_epoch=int(3 * exp_num_batches),
    )

    # multi-processed validation dataset
    mp_training_data_loader_result_03 = list(train_dataset_loader_03)

    # should contain an entry for every time series id
    transformation_counts_03 = get_transformation_counts(
        mp_training_data_loader_result_03
    )

    assert all(
        k in transformation_counts_03 for k in range(CD_NUM_TIME_SERIES)
    ), "One worker should be able to traverse all in one sweep, and should not deplete its iterator."


# This is just a general functionality test, whether the multiprocessing works in practice as expected
def test_general_functionality() -> None:
    ds_info, train_ds, test_ds = constant_dataset()
    freq = ds_info.metadata.freq
    prediction_length = ds_info.prediction_length

    ctx = "cpu"
    trainer = Trainer(ctx=ctx, epochs=3, num_batches_per_epoch=5)

    estimator = DeepAREstimator(
        prediction_length=prediction_length, freq=freq, trainer=trainer
    )

    agg_metrics, item_metrics = backtest_metrics(
        train_dataset=train_ds,
        test_dataset=test_ds,
        forecaster=estimator,
        evaluator=Evaluator(calculate_owa=False),
        num_workers=NUM_WORKERS_MP,
    )

    # just some sanity check
    assert (
        agg_metrics is not None and item_metrics is not None
    ), "Metrics should not be None if everything went smooth."


# delete cache explicitly in last test
delete_cache()
