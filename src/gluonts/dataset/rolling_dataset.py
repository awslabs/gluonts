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

from functools import partial
from typing import Optional

import numpy as np
import pandas as pd

from gluonts.dataset.common import Dataset
from gluonts.dataset.util import to_pandas


def generate_rolling_datasets(
    dataset: Dataset,
    strategy,
    start_time: pd.Timestamp,
    end_time: Optional[pd.Timestamp] = None,
) -> Dataset:
    """
    Returns an augmented version of the input dataset where each timeseries has 
    been rolled upon based on the parameters supplied. Below follows an 
    explanation and examples of how the different parameters can be used to generate
    differently rolled datasets.
    
    The 'rolling' will happen on the data available in the provided window between the 
    start_time and the end_time for each timeseries. If end_time is omitted, rolling
    happens on all datapoints from start_time until the end of the timeseries. 
    The way the data is rolled is governed by the strategy used. 

    Below examples will be based on this one timeseries long dataset
    [{
    target=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    start='2000-1-1-01'
    }]
    
    applying generate_rolling_datasets on this dataset with:

    start_time = pd.Timestamp('2000-1-1-06', '1H')
    end_time = pd.Timestamp('2000-1-1-10', '1H')
    strategy=basic_strategy(prediction_length=2)

    returns a new dataset as follows (only target values shown for brevity):

    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 6, 7]

    i.e. maximum amount of rolls possible between the end_time and start_time.
    The basic_strategy only cuts the last value of the target for as long as
    there is enough values after start_time to perform predictions on.

    When no end time is provided the output is as below since all datapoints
    from start_time will be rolled over.

    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    [1, 2, 3, 4, 5, 6, 7, 8]
    [1, 2, 3, 4, 5, 6, 7]

    If the unique_strategy is used, fewer values will be in the output as each
    roll will be of size prediction_length. This ensures that each prediction
    will be done on unique/new data. 
    
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    [1, 2, 3, 4, 5, 6, 7, 8]

    Not setting an end time and using the unique_strategy results in
    the below dataset.

    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    [1, 2, 3, 4, 5, 6, 7]

    Parameters
    ----------
    dataset
        Dataset to generate the rolling forecasting datasets from
    strategy
        The strategy that is to be used when rolling
    start_time
        The start of the window where rolling forecasts should be applied
    end_time
        The end time of the window where rolling should be applied
    Returns
    -------
    Dataset 
        The augmented dataset
    """
    assert dataset, "a dataset to perform rolling evaluation on is needed"
    assert start_time, "a pandas Timestamp object is needed for the start time"
    assert strategy, """a strategy to use when rolling is needed, consider
        using gluonts.dataset.rolling_dataset.basic_strategy"""
    if end_time:
        assert end_time > start_time, "end time has to be after the start time"

    ds = []
    for item in dataset:
        series = to_pandas(item, start_time.freq)
        base = series[:start_time][:-1].to_numpy()
        prediction_window = series[start_time:end_time]

        for window in strategy(prediction_window):
            new_item = item.copy()
            new_item["target"] = np.concatenate([base, window.to_numpy()])
            ds.append(new_item)

    return ds


def part_function(window, prediction_length, modifier):
    """
    Helper function which yields cut versions of the provided window.
    each cut is of size modifier.
    
    Parameters
    ----------
    window
        The window that should be rolled over
    prediction_length
        The prediction length of the Predictor that the dataset will be
        used with
    modifier
        The amount of data to remove for each cut
    
    Returns
    -------
    A generator which yields multiple cut versions of the window
    """
    while len(window) >= prediction_length:
        yield window
        window = window[:-modifier]


def basic_strategy(prediction_length):
    """
    Removes one datapoint for each iteration until to little data is left
    for the predictor to predict on
    
    Parameters
    ----------
    prediction_length
        The prediction length of the Predictor that the dataset will be
        used with
    
    Returns
    -------
    A partial function which yields the rolled windows
    """
    assert prediction_length, "prediction_length is needed"
    assert prediction_length > 0, "prediction length needs to be > 0"
    return partial(
        part_function, prediction_length=prediction_length, modifier=1
    )


def unique_strategy(prediction_length):
    """
    Removes datapoints equivalent to the prediction length used on each 
    iteration. Iterates until window is too small for the given 
    prediction_length.
    
    Parameters
    ----------
    prediction_length
        The prediction length of the Predictor that the dataset will be
        used with
    
    Returns
    -------
    A partial function which yields the rolled windows
    """
    assert prediction_length, "prediction_length is needed"
    assert prediction_length > 0, "prediction length needs to be > 0"
    return partial(
        part_function,
        prediction_length=prediction_length,
        modifier=prediction_length,
    )
