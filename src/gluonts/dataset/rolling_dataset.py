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

from typing import Optional

import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from pydantic import BaseModel


class StepStrategy(BaseModel):
    """
    Removes datapoints equivalent to step_size for each iteration until
    amount of data left is less than prediction_length

    Parameters
    ----------
    prediction_length
        The prediction length of the Predictor that the dataset will be
        used with
    step_size
        The number of points to remove for each iteration.
    """

    prediction_length: int
    step_size: int = 1

    def get_windows(self, window):
        """
        This function splits a given window (array of target values) into
        smaller chunks based on the provided parameters of the parent class.

        Parameters
        ----------
        window
            The window which should be split

        Returns
        ----------
        A generator yielding split versions of the window
        """
        assert (
            self.prediction_length > 0
        ), """the step strategy requires a prediction_length > 0"""
        assert self.step_size > 0, """step_size should be > 0"""

        while len(window) >= self.prediction_length:
            yield window
            window = window[: -self.step_size]


class NumSplitsStrategy(BaseModel):
    """
    The NumSplitsStrategy splits a window into *num_splits* chunks of equal size.

    Parameters
    ----------
    prediction_length
        The prediction length of the Predictor that the dataset will be
        used with
    num_splits
        The number of segments which the window should be split into
    """

    prediction_length: int
    num_splits: int

    def get_windows(self, window):
        """
        This function splits a given window (array of target values) into
        smaller chunks based on the provided parameters of the parent class.

        Parameters
        ----------
        window
            The window which should be split

        Returns
        ----------
        A generator yielding split versions of the window
        """
        assert self.num_splits > 1, """num_splits should be > 1"""
        for slice_idx in np.linspace(
            start=self.prediction_length, stop=len(window), num=self.num_splits
        ):

            yield window[: int(round(slice_idx))]


def truncate_features(timeseries: dict, max_len: int) -> dict:
    """truncate dynamic features to match `max_len` length"""
    for key in (
        FieldName.FEAT_DYNAMIC_CAT,
        FieldName.FEAT_DYNAMIC_REAL,
    ):
        if not key in timeseries:
            continue
        timeseries[key] = [feature[:max_len] for feature in timeseries[key]]

    return timeseries


# TODO Add parameter allowing for rolling of other arrays
# with a time axis such as time dependent features
def generate_rolling_dataset(
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

    The *rolling* happens on the data available in the provided window between the
    *start_time* and the *end_time* for each timeseries. If *end_time* is omitted, rolling
    happens on all datapoints from *start_time* until the end of the timeseries.
    The way the data is rolled is governed by the strategy used.

    Below examples will be based on this one timeseries long dataset

    >>> ds = [{
    ...     "target": np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    ...     "start": pd.Timestamp('2000-1-1-01', freq='1H')
    ... }]

    applying generate_rolling_dataset on this dataset like:

    >>> rolled = generate_rolling_dataset(
    ...     dataset=ds,
    ...     strategy = StepStrategy(prediction_length=2),
    ...     start_time = pd.Timestamp('2000-1-1-06', '1H'),
    ...     end_time = pd.Timestamp('2000-1-1-10', '1H')
    ... )

    Results in a new dataset as follows (only target values shown for brevity):

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9]\n
        [1, 2, 3, 4, 5, 6, 7, 8]\n
        [1, 2, 3, 4, 5, 6, 7]\n

    i.e. maximum amount of rolls possible between the *end_time* and *start_time*.
    The StepStrategy only cuts the last value of the target for as long as
    there is enough values after *start_time* to perform predictions on.

    When no end time is provided the output is as below since all datapoints
    from *start_time* will be rolled over.

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9]\n
        [1, 2, 3, 4, 5, 6, 7, 8]\n
        [1, 2, 3, 4, 5, 6, 7]

    One can change the step_size of the strategy as below:

    >>> strategy = StepStrategy(prediction_length=2, step_size=2)


    This causes fewer values to be in the output which,
    when prediction_length matches step_size, ensures that each prediction
    will be done on unique/new data. Below is the output when the above strategy is used.

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n
        [1, 2, 3, 4, 5, 6, 7, 8]

    Not setting an end time and using the step_size=2 results in
    the below dataset.

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]\n
        [1, 2, 3, 4, 5, 6, 7, 8, 9]\n
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
    ----------
    Dataset
        The augmented dataset


    """
    assert dataset, "a dataset to perform rolling evaluation on is needed"
    assert start_time, "a pandas Timestamp object is needed for the start time"
    assert strategy, """a strategy to use when rolling is needed, for example
        gluonts.dataset.rolling_dataset.StepStrategy"""
    if end_time:
        assert end_time > start_time, "end time has to be after the start time"

    ds = []
    for item in dataset:
        series = to_pandas(item, start_time.freq)
        base = series[:start_time][:-1].to_numpy()
        prediction_window = series[start_time:end_time]

        for window in strategy.get_windows(prediction_window):
            new_item = item.copy()
            new_item[FieldName.TARGET] = np.concatenate(
                [base, window.to_numpy()]
            )
            new_item = truncate_features(
                new_item, len(new_item[FieldName.TARGET])
            )
            ds.append(new_item)

    return ds
