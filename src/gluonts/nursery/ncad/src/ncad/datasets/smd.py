"""
More details about the data can be found in:
https://github.com/NetManAIOps/OmniAnomaly
"""

import os

from typing import Union
from pathlib import Path, PosixPath

import numpy as np

import re

from ncad.ts import TimeSeries, TimeSeriesDataset

import tqdm


def smd(path: Union[PosixPath, str], _for_testing=False, *args, **kwargs) -> TimeSeriesDataset:
    """

    Args:
        path : Path to the directory containing the four benchmarks and their corresponding csv files.
        _for_testing : Internal boolean argument, if True, only loads two time series.

    Source:
        https://github.com/NetManAIOps/OmniAnomaly
    """
    print("\nLoading SMD dataset...\n")

    path = PosixPath(path).expanduser()
    assert path.is_dir()

    # Verify that all subdirectories exist
    bmk_dirs = ["train", "test", "test_label"]
    assert np.all([(path / bmk_dir).is_dir() for bmk_dir in bmk_dirs])

    # Files to be read
    train_files = [fn for fn in os.listdir(path / "train") if fn.endswith(".txt")]
    train_files.sort()
    test_files = [fn for fn in os.listdir(path / "test") if fn.endswith(".txt")]
    test_files.sort()
    test_label_files = [fn for fn in os.listdir(path / "test") if fn.endswith(".txt")]
    test_label_files.sort()

    # Check that train and test files have the same names
    assert train_files == test_files
    # Check that all labels files are available
    assert test_files == test_label_files

    train_dataset = TimeSeriesDataset()
    test_dataset = TimeSeriesDataset()

    if _for_testing:
        # For some testing, only load two TimeSeries
        train_files = train_files[:2]

    for fn_i in tqdm.tqdm(train_files):

        ts_id = re.sub(".txt$", "", fn_i)

        # Load the multivariate time series from txt files
        ts_train_np, ts_test_np, test_anomalies = [
            np.genfromtxt(
                fname=path / dir_j / fn_i,
                dtype=np.float32,
                delimiter=",",
            )
            for dir_j in bmk_dirs
        ]
        train_anomalies = None

        train_dataset.append(
            TimeSeries(
                values=ts_train_np,
                labels=train_anomalies,
                item_id=f"{ts_id}_train",
            )
        )
        test_dataset.append(
            TimeSeries(
                values=ts_test_np,
                labels=test_anomalies,
                item_id=f"{ts_id}_test",
            )
        )

    print("\n...SMD dataset loaded succesfully.\n")

    return train_dataset, test_dataset
