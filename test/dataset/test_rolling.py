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

from math import floor
from gluonts.evaluation.backtest import generate_rolling_datasets
import pandas as pd
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.common import ListDataset
import pytest


len_to_truncate = 5
length_of_roll_window = 5


def generate_expected_dataset_unique(prediction_length):
    """
    test set for the variation that behaves like:

    [1,2,3,4,5,6,7,8,9] 
             ^
        start_time
    
    becomes
    
    [1,2,3,4,5,6,7,8,9]
    [1,2,3,4,5,6,7]
    
    for 
    
    length_of_roll_window = 5
    prediction_length = 2
    """
    # for constant dataset
    length_timeseries = 30
    num_timeseries = 10
    a = []
    trunc_length = length_timeseries - len_to_truncate
    for i in range(num_timeseries):
        iterations = floor(length_of_roll_window / prediction_length)
        for ii in range(iterations):
            a.append((float(i), trunc_length - prediction_length * ii))

    return a


def generate_expected_dataset_standard(prediction_length):
    """
    test set for the variation that behaves like:

    [1,2,3,4,5,6,7,8,9] 
             ^
        start_time
    becomes
    
    [1,2,3,4,5,6,7,8,9]
    [1,2,3,4,5,6,7,8]
    [1,2,3,4,5,6,7]
    [1,2,3,4,5,6]
    
    for 
    
    length_of_roll_window = 5
    prediction_length = 2
    """
    # for constant dataset
    length_timeseries = 30
    num_timeseries = 10

    a = []
    trunc_length = length_timeseries - len_to_truncate
    for i in range(num_timeseries):
        for ii in range(length_of_roll_window - prediction_length + 1):
            a.append((float(i), trunc_length - ii))
    return a


def generate_expected_dataset_varying_standard(prediction_length):
    ds_list = None
    if prediction_length == 2:
        # tests 4,5,8,9 should all return empty timeseries, thus they are
        # not part of the expected dataset
        lengths = [
            25,  # start test 1
            24,
            23,
            22,
            25,  # start test 2
            24,
            23,
            22,
            23,  # start test 3
            22,
            5,  # start test 6
            4,
            3,
            2,
            3,  # start test 7
            2,
            3,  # start test 10
            2,
        ]
        ds_list = [(float(0), length) for length in lengths]
    return ds_list


def generate_expected_dataset_varying_unique(prediction_length):
    ds_list = None
    if prediction_length == 2:
        # tests 4,5,8,9 should all return empty timeseries, thus they are
        # not part of the expected dataset
        lengths = [
            25,  # start test 1
            23,
            25,  # start test 2
            23,
            23,  # start test 3
            5,  # start test 6
            3,
            3,  # start test 7
            3,  # start test 10
        ]
        ds_list = [(float(0), length) for length in lengths]
    return ds_list


def check_target_values(ds, to_compare):
    i = 0
    for ts in ds:
        assert (
            len(ts["target"]) == to_compare[i][1]
        ), "timeseries {} failed".format(i + 1)
        for val in ts["target"]:
            assert val == to_compare[i][0]
        i = i + 1

    assert len(to_compare) == i


def generate_expected_rolled_dataset(pl, unique_rolls, ds_name):
    to_compare = None

    if unique_rolls and ds_name == "constant":
        to_compare = generate_expected_dataset_unique(pl)
    elif ds_name == "constant":
        to_compare = generate_expected_dataset_standard(pl)
    elif unique_rolls and ds_name == "varying":
        to_compare = generate_expected_dataset_varying_unique(pl)
    elif ds_name == "varying":
        to_compare = generate_expected_dataset_varying_standard(pl)

    assert to_compare, "dataset to compare with is not implemented"

    return to_compare


def generate_dataset(name):
    dataset = None
    if name == "constant":
        _, _, dataset = constant_dataset()
    elif name == "varying":
        # Tests edge cases
        # t0: start time of target
        # ts: start time of desired range
        # te: end time of desired range
        # t1: end time of target
        # ts < te, t0 <= t1
        #
        # start time index of rolling window is 20
        # end time index of rolling window is 24
        # ts = 2000-01-01 20:00:00
        # te = 2000-01-02 00:00:00

        f = "H"
        ds_list = [
            {  # test 1: target ends after end time, te > t1
                "target": [0.0 for i in range(30)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 2: target ends at the end time, te == t1
                "target": [0.0 for i in range(25)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 3: target ends between start and end times, ts < t1 < te
                "target": [0.0 for i in range(23)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 4: target ends on start time, ts == t1
                "target": [0.0 for i in range(20)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 5: target ends before start time, t1 < ts
                "target": [0.0 for i in range(15)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 6: target starts on start ends after end, ts == t0, te > t1
                "target": [0.0 for i in range(10)],
                "start": pd.Timestamp(2000, 1, 1, 20, 0),
            },
            {  # test 7: starts in between start time and end, ts < t0 < te < t1
                "target": [0.0 for i in range(10)],
                "start": pd.Timestamp(2000, 1, 1, 22, 0),
            },
            {  # test 8: target starts on end time, te == t0
                "target": [0.0 for i in range(10)],
                "start": pd.Timestamp(2000, 1, 2, 0, 0),
            },
            {  # test 9: starts after end time, te < t0
                "target": [0.0 for i in range(10)],
                "start": pd.Timestamp(2000, 1, 2, 1, 0),
            },
            {  # test 10: starts after ts and ends before te, ts < t0 < t1 < te
                "target": [0.0 for i in range(3)],
                "start": pd.Timestamp(2000, 1, 1, 21, 0),
            },
        ]
        dataset = ListDataset(ds_list, f)
    else:
        raise ValueError
    return dataset


@pytest.mark.parametrize(
    "prediction_length, unique",
    [(p, u) for p in [-1, 0, 5] for u in [True, False]],
)
def test_fails(prediction_length, unique):
    try:
        generate_rolling_datasets(
            dataset=generate_dataset("constant"),
            prediction_length=prediction_length,
            start_time=pd.Timestamp("2000-01-01-20", freq="1H"),
            end_time=pd.Timestamp("2000-01-02-00", freq="1H"),
            use_unique_rolls=unique,
        )
        # program should have failed at this point
        raise RuntimeWarning
    except AssertionError:
        pass


@pytest.mark.parametrize(
    "ds_name, prediction_length, unique",
    [("varying", 2, False), ("varying", 2, True)]  # mostly tests edge cases
    + [  # tests basic functionality
        ("constant", prediction_length, unique)
        for prediction_length in range(length_of_roll_window)
        for unique in [True, False]
    ],
)
def test_successes(ds_name, prediction_length, unique):
    # prediction_length=0 should fail and is handled in test_fails(...)
    if prediction_length == 0:
        return

    rolled_ds = generate_rolling_datasets(
        dataset=generate_dataset(ds_name),
        prediction_length=prediction_length,
        start_time=pd.Timestamp("2000-01-01-20", freq="1H"),
        end_time=pd.Timestamp("2000-01-02-00", freq="1H"),
        use_unique_rolls=unique,
    )

    ds_expected = generate_expected_rolled_dataset(
        prediction_length, unique, ds_name
    )

    check_target_values(rolled_ds, ds_expected)
