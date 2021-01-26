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

import json
import multiprocessing as mp
import random
import tempfile
import time
from collections import defaultdict
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from mxnet.context import current_context

from gluonts.dataset.artificial import ConstantDataset, constant_dataset
from gluonts.dataset.common import FileDataset, ListDataset

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    InferenceDataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.batchify import batchify
from gluonts.mx.trainer import Trainer
from gluonts.transform import (
    Chain,
    InstanceSampler,
    InstanceSplitter,
    UniformSplitSampler,
)

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
            stack_fn=partial(batchify, ctx=current_context()),
            num_workers=None,  # This is the crucial difference
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

    validation_dataset_loader = ValidationDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        stack_fn=partial(batchify, ctx=current_context()),
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
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
        mp_val_data_loader_result_02[0]["past_target"].context
        == current_context()
    ), "Batches in incorrect context"


@flaky(max_runs=5, min_passes=1)
@pytest.mark.parametrize(
    "num_workers",
    [
        i
        for i in [
            None,
            1,
            2,
        ]
        if i is None or i <= mp.cpu_count()
    ],
    # TODO: using more than 2 is a problem for our tests, if some of the cores are busy and fall behind
    # TODO: using multiple input queues in the loader would make this pass no matter how busy each core is
    # [i for i in [None, 1, 2, 3, 4] if i is None or i <= mp.cpu_count()],
)
def test_train_loader_goes_over_all_data(num_workers) -> None:
    batch_size = 4
    num_batches_per_epoch = 4
    num_time_series = batch_size * num_batches_per_epoch * 3
    num_passes = 5
    num_epochs = num_passes * 3

    simple_data = [
        {
            "start": "2012-01-01",
            "target": np.random.uniform(size=40).astype(float).tolist(),
            "item_id": i,
        }
        for i in range(num_time_series)
    ]

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
            stack_fn=partial(batchify, ctx=current_context()),
            num_workers=num_workers,
        )

        item_ids = defaultdict(int)

        for epoch in range(num_epochs):
            for batch in islice(dl, num_batches_per_epoch):
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

    # original no multiprocessing processed validation dataset
    inference_loader_data_transformed_original = list(
        InferenceDataLoader(
            dataset=list_dataset,
            transform=transformation,
            batch_size=BATCH_SIZE,
            stack_fn=partial(batchify, ctx=current_context()),
            num_workers=None,  # This is the crucial difference
        )
    )

    inference_loader = InferenceDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        stack_fn=partial(batchify, ctx=current_context()),
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
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
        mp_inf_data_loader_result_02[0]["past_target"].context
        == current_context()
    ), "Batches in incorrect context"


# Batches of the train data loader can only be of the same exact desired size
# Unlike the inference or validation data loader, which can have varying batch sizes, if the number
# of time series is not divisible by BATCH_SIZE * NUM_WORKERS_MP.
def test_training_loader_batch_size_hard_constraint() -> None:
    (
        list_dataset,
        transformation,
        list_dataset_pred_length,
        train_data_transformed_original,
    ) = get_dataset_and_transformation()

    train_dataset_loader_1 = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        stack_fn=partial(batchify, ctx=current_context()),
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
    )

    train_dataset_loader_2 = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        stack_fn=partial(batchify, ctx=current_context()),
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
        shuffle_buffer_length=3 * BATCH_SIZE,
    )

    batches_1 = list(islice(train_dataset_loader_1, 30))
    batches_2 = list(islice(train_dataset_loader_2, 30))

    assert all(
        [len(batch["item_id"]) == BATCH_SIZE for batch in batches_1]
    ), "Not every batch from training loader is right size."

    assert all(
        [len(batch["item_id"]) == BATCH_SIZE for batch in batches_2]
    ), "Not every batch from training loader is right size, with shuffling on."


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

    train_dataset_loader = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        stack_fn=partial(batchify, ctx=current_context()),
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
    )

    batches = list(islice(train_dataset_loader, int(3 * exp_num_batches)))
    transformation_counts = get_transformation_counts(batches)

    assert all(
        [k in transformation_counts for k in range(CD_NUM_TIME_SERIES)]
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

    train_dataset_loader = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        stack_fn=partial(batchify, ctx=current_context()),
        num_workers=NUM_WORKERS_MP,  # This is the crucial difference
    )

    batches = list(islice(train_dataset_loader, int(0.5 * exp_num_batches)))
    transformation_counts = get_transformation_counts(batches)

    assert not all(
        [k in transformation_counts for k in range(CD_NUM_TIME_SERIES)]
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

    train_dataset_loader = TrainDataLoader(
        dataset=list_dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        stack_fn=partial(batchify, ctx=current_context()),
        num_workers=1,  # This is the crucial difference
    )

    batches = list(islice(train_dataset_loader, int(3 * exp_num_batches)))
    transformation_counts = get_transformation_counts(batches)

    assert all(
        k in transformation_counts for k in range(CD_NUM_TIME_SERIES)
    ), "One worker should be able to traverse all in one sweep, and should not deplete its iterator."


# This is just a general functionality test, whether the multiprocessing works in practice as expected
def test_general_functionality() -> None:
    ds_info, train_ds, test_ds = constant_dataset()
    freq = ds_info.metadata.freq
    prediction_length = ds_info.prediction_length

    trainer = Trainer(epochs=3, num_batches_per_epoch=5)

    estimator = DeepAREstimator(
        prediction_length=prediction_length, freq=freq, trainer=trainer
    )

    predictor = estimator.train(training_data=train_ds)

    agg_metrics, item_metrics = backtest_metrics(
        test_dataset=test_ds,
        predictor=predictor,
        evaluator=Evaluator(calculate_owa=False),
    )

    # just some sanity check
    assert (
        agg_metrics is not None and item_metrics is not None
    ), "Metrics should not be None if everything went smooth."


# delete cache explicitly in last test
delete_cache()
