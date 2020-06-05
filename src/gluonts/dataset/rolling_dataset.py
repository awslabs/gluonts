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

    assert dataset, "a dataset to perform rolling evaluation on is needed"
    assert start_time, "a pandas Timestamp object is needed for the start time"
    assert strategy, '''a strategy to use when rolling is needed, consider
        using gluonts.dataset.rolling_dataset.basic_strategy'''
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
    while len(window) >= prediction_length:
        yield window
        window = window[:-modifier]


def basic_strategy(prediction_length):
    assert prediction_length, "prediction_length is needed"
    assert prediction_length > 0, "prediction length needs to be > 0"
    return partial(part_function, prediction_length=prediction_length, modifier=1)


def unique_strategy(prediction_length):
    assert prediction_length, "prediction_length is needed"
    assert prediction_length > 0, "prediction length needs to be > 0"
    return partial(part_function, prediction_length=prediction_length, modifier=prediction_length)

