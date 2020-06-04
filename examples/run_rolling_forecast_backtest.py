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
from gluonts.dataset.artificial import default_synthetic, constant_dataset
from gluonts.dataset.common import ListDataset

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


# TODO implement this
def generate_expected_dataset_varying_unique(prediction_length):
    return []


def test_targets(ds, to_compare):
    i = 0
    for ts in ds:
        # print('real:', len(ts['target']), 'reference:', to_compare[i][1])
        assert (
            len(ts["target"]) == to_compare[i][1]
        ), "timeseries {} failed".format(i + 1)
        for val in ts["target"]:
            # print('real:', val,'reference:', to_compare[i][0])
            assert val == to_compare[i][0]
        i = i + 1

    assert len(to_compare) == i
    # print('Tested timeseries', i,'\n')
    # print("dataset passes target length and value checks\n")


def generate_expected_rolled_dataset(pl, unique_rolls, ds_name):
    to_compare = None

    # print(
    #    ' unique_rolls:\t\t',unique_rolls,'\n',
    #    'prediction_length:\t', pl, '\n',
    #    'dataset name:\t\t', ds_name)

    if unique_rolls and ds_name == "constant":
        to_compare = generate_expected_dataset_unique(pl)
    elif ds_name == "constant":
        to_compare = generate_expected_dataset_standard(pl)
    elif unique_rolls and ds_name == "varying":
        to_compare = generate_expected_dataset_varying_unique(pl)
    elif ds_name == "varying":
        to_compare = generate_expected_dataset_varying_standard(pl)
    return to_compare


def generate_dataset(name):
    dataset = None
    if name == "constant":
        _, _, dataset = constant_dataset()
    elif name == "varying":
        f = "H"
        # start of target is 2000-01-01 00:00:00
        # start time is 2000-01-01 20:00:00
        # end time is 2000-01-02 00:00:00
        # end time of target is 2000-01-02 05:00:00
        # start time index of rolling window is 20
        # end time index of rolling window is 24
        ds_list = [
            {  # t1: ends after end time
                "target": [0.0 for i in range(30)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # t2: ends at the end time
                "target": [0.0 for i in range(25)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # t3: ends in between start and end times
                "target": [0.0 for i in range(23)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # t4: end in the beginning of start time
                "target": [0.0 for i in range(20)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # t5: ends before start time
                "target": [0.0 for i in range(15)],
                "start": pd.Timestamp(2000, 1, 1, 0, 0),
            },
            {  # t6: starts on start time
                "target": [0.0 for i in range(10)],
                "start": pd.Timestamp(2000, 1, 1, 20, 0),
            },
            {  # t7: starts in between start time and end
                "target": [0.0 for i in range(10)],
                "start": pd.Timestamp(2000, 1, 1, 22, 0),
            },
            {  # t8: starts on end time
                "target": [0.0 for i in range(10)],
                "start": pd.Timestamp(2000, 1, 2, 0, 0),
            },
            {  # t9: starts after end time
                "target": [0.0 for i in range(10)],
                "start": pd.Timestamp(2000, 1, 2, 1, 0),
            },
            {  # t10: starts after start time and ends before end time
                "target": [0.0 for i in range(3)],
                "start": pd.Timestamp(2000, 1, 1, 21, 0),
            },
        ]
        dataset = ListDataset(ds_list, f)
    else:
        print("unknown dataset name")
        exit(1)
    return dataset


def get_times():
    # returns the times that rolling forecasts should be applied on
    # for this dataset the values that should be taken into account is:
    # [20, 21, 22, 23, 24]
    t_start = pd.Timestamp(2000, 1, 1, 0, 0)
    period = pd.date_range(start=t_start, periods=30, freq="H")
    t0 = period[20]
    # print(t0)
    t1 = period[24]
    # print(t1)
    return t0, t1


def test_fails():
    dataset = generate_dataset("constant")
    t0, t1 = get_times()

    # test the inputs that should fail
    inputs = [
        (-1, True),
        (-1, False),
        (0, True),
        (0, False),
        (5, True),
        (5, False),
    ]

    for i in inputs:
        try:
            generate_rolling_datasets(
                dataset=dataset,
                prediction_length=i[0],
                start_time=t0,
                end_time=t1,
                use_unique_rolls=i[1],
            )
        except AssertionError:
            pass
            # print('dataset fails to be generated as expected on:', *i)
        except:
            print(
                """generate rolling dataset passes when supposed to fail on:""",
                *i
            )
    print("Tests that were supposed to fail, failed successfully")


def test_successes():
    # test the inputs that should succeed
    datasets_to_test = ["varying", "constant"]
    t0, t1 = get_times()
    for ds_name in datasets_to_test:
        dataset = generate_dataset(ds_name)
        for i in range(length_of_roll_window):

            # i=0 should fail
            if i == 0:
                continue

            for unique in [True, False]:
                ds = generate_rolling_datasets(
                    dataset=dataset,
                    prediction_length=i,
                    start_time=t0,
                    end_time=t1,
                    use_unique_rolls=unique,
                )

                ds_expected = generate_expected_rolled_dataset(
                    i, unique, ds_name
                )
                if ds_expected:
                    test_targets(ds, ds_expected)

    print("all inputs that that should pass, passed")


if __name__ == "__main__":
    test_fails()
    test_successes()

    dataset = get_dataset("constant", regenerate=False)

    estimator = SimpleFeedForwardEstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        trainer=Trainer(epochs=5, num_batches_per_epoch=10),
    )

    predictor = estimator.train(dataset.train)

    dataset_rolled = generate_rolling_datasets(
        dataset=dataset.test,
        prediction_length=dataset.metadata.prediction_length,
        start_time=pd.Timestamp("2000-01-01-15", freq="1H"),
        end_time=pd.Timestamp("2000-01-02-04", freq="1H"),
        use_unique_rolls=True,
    )

    for ds in [dataset.test, dataset_rolled]:
        forecast_it, ts_it = make_evaluation_predictions(
            ds, predictor=predictor, num_samples=len(ds)
        )

        agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it)

        pprint.pprint(agg_metrics)
        print("\n")
