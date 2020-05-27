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

"""
This example shows how to fit a model and evaluate its predictions.
"""
import pprint
from math import floor
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import (
    make_evaluation_predictions,
    generate_rolling_datasets,
)
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
import pandas as pd
from gluonts.dataset.common import TimeSeriesItem


def test_dataset_lengths(datasets):
    ds_eval = iter(datasets.to_evaluate)
    ds_pred = iter(datasets.to_predict)

    c = True
    ts1 = ts2 = None
    while c:
        try:
            ts1 = next(ds_pred)
        except:
            ts1 = None

        try:
            ts2 = next(ds_eval)
        except:
            ts2 = None

        assert type(ts1) == type(ts2)
        c = (ts1 == None) or (ts2 == None)
    print("dataset lengths match eachother\n")


def test_ds_eval_targets(datasets):
    ds_eval = iter(datasets.to_evaluate)
    for ts in ds_eval:
        tss = ts["target"]
        assert len(tss) == 25

    print("ds_eval passes target length check\n")


def generate_expected_predict():
    a = []
    for i in range(10):
        for ii in range(5):
            a.append((float(i), 25 - ii - 1))

    return a


def test_ds_pred_targets(datasets):
    ds_pred = iter(datasets.to_predict)

    to_compare = generate_expected_predict()
    i = 0
    for ts in ds_pred:
        # print(len(ts['target']), to_compare[i][1])
        assert len(ts["target"]) == to_compare[i][1]
        for val in ts["target"]:
            # print(val, to_compare[i][0])
            assert val == to_compare[i][0]
        i = i + 1
    print("ds_pred passes target length and value checks\n")


def generate_datasets():
    dataset = get_dataset("constant", regenerate=False)
    """
    estimator = SimpleFeedForwardEstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        trainer=Trainer(epochs=5, num_batches_per_epoch=10),
    )
    """
    # predictor = estimator.train(dataset.train)

    # calculate time points of test set
    ts = next(iter(dataset.test))

    length = len(ts["target"])  # 30
    len_to_truncate = 5
    length_of_roll = 5
    test_end = length - len_to_truncate  # 25
    # we add one as 21, 22, 23, 24 and 25 will be the total items that will be in the rolling forecast
    test_start = test_end - length_of_roll + 1  # 21

    print("test_start: {}\ntest_end: {}".format(test_start, test_end))

    # Calculate Timestamps of test range
    t = ts["start"]
    f = t.freq
    t_end = t + f * (length - 1)
    t0 = t + f * (test_start - 1)
    t1 = t + f * (test_end - 1)

    t_index = 0
    t0_index = 20
    t1_index = 24
    t_end_index = 29

    print("start: {}\nt0: {}\nt1: {}\nt_end: {}".format(t, t0, t1, t_end))
    print(
        "start: {}\nt0: {}\nt1: {}\nt_end: {}".format(
            t_index, t0_index, t1_index, t_end_index
        )
    )

    assert t0_index == len(pd.date_range(start=t, end=t0, freq=f)) - 1
    assert t1_index == len(pd.date_range(start=t, end=t1, freq=f)) - 1
    assert t_end_index == len(pd.date_range(start=t, end=t_end, freq=f)) - 1

    # t0 is start of window to perform rolling forecasts on
    # t1 is end of window to perform rolling forecasts on

    datasets = generate_rolling_datasets(dataset.test, 1, t0, t1)
    return datasets


if __name__ == "__main__":

    # test so that length of evaluation dataset is consistent with the predict dataset
    test_dataset_lengths(generate_datasets())
    test_ds_eval_targets(generate_datasets())
    test_ds_pred_targets(generate_datasets())
