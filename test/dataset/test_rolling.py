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

# third party imports
import pandas as pd
import pytest

# first party imports
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset,
)


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

        ds_list = [
            {  # test 1: ends after end time, te > t1
                "target": [0.0] * 30,
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 2: ends at the end time, te == t1
                "target": [0.0] * 25,
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 3: ends between start and end times, ts < t1 < te
                "target": [0.0] * 23,
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 4: ends on start time, ts == t1
                "target": [0.0] * 20,
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 5: ends before start time, t1 < ts
                "target": [0.0] * 15,
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # test 6: starts on start ends after end, ts == t0, te > t1
                "target": [0.0] * 10,
                "start": pd.Timestamp(2000, 1, 1, 20, 0),
            },
            {  # test 7: starts in between ts and te, ts < t0 < te < t1
                "target": [0.0] * 10,
                "start": pd.Timestamp(2000, 1, 1, 22, 0),
            },
            {  # test 8: starts on end time, te == t0
                "target": [0.0] * 10,
                "start": pd.Timestamp(2000, 1, 2, 0, 0),
            },
            {  # test 9: starts after end time, te < t0
                "target": [0.0] * 10,
                "start": pd.Timestamp(2000, 1, 2, 1, 0),
            },
            {  # test 10: starts after ts & ends before te, ts < t0 < t1 < te
                "target": [0.0] * 3,
                "start": pd.Timestamp(2000, 1, 1, 21, 0),
            },
        ]
        dataset = ListDataset(ds_list, "H")
    else:
        pytest.raises(ValueError)
    return dataset


@pytest.mark.parametrize(
    "prediction_length, unique",
    [(p, u) for p in [-1, 0] for u in [True, False]],
)
def test_invalid_rolling_parameters(prediction_length, unique):
    try:
        generate_rolling_dataset(
            dataset=generate_dataset("constant"),
            start_time=pd.Timestamp("2000-01-01-20", freq="1H"),
            end_time=pd.Timestamp("2000-01-02-00", freq="1H"),
            strategy=StepStrategy(
                step_size=prediction_length if unique else 1,
                prediction_length=prediction_length,
            ),
        )
        # program should have failed at this point
        pytest.raises(RuntimeError)
    except AssertionError:
        pass


@pytest.mark.parametrize(
    "ds_name, prediction_length, unique, ignore_end, ds_expected",
    [
        (
            "varying",
            2,
            False,
            False,
            [[0.0] * length for length in range(25, 21, -1)] * 2
            + [[0.0] * length for length in [23, 22]]
            + [[0.0] * length for length in range(5, 1, -1)]
            + [[0.0] * length for length in [3, 2]] * 2,
        ),
        (
            "varying",
            2,
            True,
            False,
            [
                [0.0] * 25,
                [0.0] * 23,
                [0.0] * 25,
                [0.0] * 23,
                [0.0] * 23,
                [0.0] * 5,
                [0.0] * 3,
                [0.0] * 3,
                [0.0] * 3,
            ],
        ),
        (
            "varying",
            3,
            False,
            True,
            [[0.0] * length for length in range(30, 22, -1)]
            + [[0.0] * length for length in [25, 24, 23]]
            + [[0.0] * 23]
            + [[0.0] * length for length in range(10, 2, -1)] * 4
            + [[0.0] * 3],
        ),
        (
            "constant",
            1,
            True,
            False,
            [
                [float(val)] * length
                for val in range(10)
                for length in [25, 24, 23, 22, 21]
            ],
        ),
        (
            "constant",
            1,
            False,
            False,
            [
                [float(val)] * length
                for val in range(10)
                for length in [25, 24, 23, 22, 21]
            ],
        ),
        (
            "constant",
            2,
            True,
            False,
            [
                [float(val)] * length
                for val in range(10)
                for length in [25, 23]
            ],
        ),
        (
            "constant",
            2,
            False,
            False,
            [
                [float(val)] * length
                for val in range(10)
                for length in [25, 24, 23, 22]
            ],
        ),
        (
            "constant",
            3,
            True,
            False,
            [[float(val)] * 25 for val in range(10)],
        ),
        (
            "constant",
            3,
            False,
            False,
            [
                [float(val)] * length
                for val in range(10)
                for length in [25, 24, 23]
            ],
        ),
        (
            "constant",
            4,
            True,
            False,
            [[float(val)] * 25 for val in range(10)],
        ),
        (
            "constant",
            4,
            False,
            False,
            [
                [float(val)] * length
                for val in range(10)
                for length in [25, 24]
            ],
        ),
    ],
)
def test_step_strategy(
    ds_name, prediction_length, unique, ignore_end, ds_expected
):
    rolled_ds = generate_rolling_dataset(
        dataset=generate_dataset(ds_name),
        start_time=pd.Timestamp("2000-01-01-20", freq="1H"),
        end_time=None
        if ignore_end
        else pd.Timestamp("2000-01-02-00", freq="1H"),
        strategy=StepStrategy(
            step_size=prediction_length if unique else 1,
            prediction_length=prediction_length,
        ),
    )

    i = 0
    for ts in rolled_ds:
        assert len(ts["target"]) == len(ds_expected[i])
        for rolled_result, expected in zip(ts["target"], ds_expected[i]):
            assert rolled_result == expected
        i += 1

    assert len(ds_expected) == i
