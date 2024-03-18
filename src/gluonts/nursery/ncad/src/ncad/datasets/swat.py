"""
More details about the data can be found in:
https://github.com/NetManAIOps/OmniAnomaly
"""

import os

from typing import Union
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd

from ncad.ts import TimeSeries, TimeSeriesDataset


def swat(
    path: Union[PosixPath, str],
    subsample_one_in_k: int = 10,
    subsample_fun: str = ["mean", "median"][0],
    multivariate: bool = True,
    *args,
    **kwargs,
) -> TimeSeriesDataset:
    """

    Args:
        path : Path to the directory containing the four benchmarks and their corresponding csv files.

    Source:
        iTrust Labs (https://itrust.sutd.edu.sg/)
    """

    print("\nLoading SWaT dataset...\n")

    path = PosixPath(path).expanduser()
    assert path.is_dir()

    # Verify that all files exist
    files_swat = ["SWaT_Dataset_Normal_v0.csv", "SWaT_Dataset_Attack_v0.csv"]
    assert np.all([fn in os.listdir(path) for fn in files_swat])

    # Load csv files
    # NOTE: Data preprocessing
    #   -Eliminate row above column names
    #   -Change first column name " Timestamp" to "Timestamp"
    #   -Rename "Normal/Attack" to "Label", and recode "Normal" -> 0, "Attack" -> 1, "A ttack" -> 1,
    #   -save as csv
    train_df, test_df = [pd.read_csv(path / file) for file in files_swat]

    assert train_df.shape == (
        496800,
        53,
    ), f"Problem with 'SWaT_Dataset_Normal_v0.csv', expected shape (449919, 53), loaded shape {train_df.shape}"
    assert test_df.shape == (
        449919,
        53,
    ), f"Problem with 'SWaT_Dataset_Attack_v0.csv', expected shape (449919, 53), loaded shape {test_df.shape}"
    assert all([col in train_df.columns for col in ["Timestamp", "Label"]])
    assert all([col in test_df.columns for col in ["Timestamp", "Label"]])
    assert all(
        train_df["Label"] == 0
    ), "No anomalies were expected in 'SWaT_Dataset_Normal_v0.csv', but some where found"

    train_time = train_df.pop("Timestamp")
    train_label = train_df.pop("Label")
    test_time = test_df.pop("Timestamp")
    test_label = test_df.pop("Label")

    # Define subsample functions
    subsample_labels = lambda df, k: df.copy().groupby(np.arange(len(df)) // k).max()
    if subsample_fun == "mean":
        subsample_values = lambda df, k: df.copy().groupby(np.arange(len(df)) // k).mean()
    elif subsample_fun == "median":
        subsample_values = lambda df, k: df.copy().groupby(np.arange(len(df)) // k).median()

    # Subsample training data
    train_df = subsample_values(train_df, subsample_one_in_k)
    train_label = subsample_labels(train_label, subsample_one_in_k)
    # Subsample test data
    test_df = subsample_values(test_df, subsample_one_in_k)
    test_label = subsample_labels(test_label, subsample_one_in_k)

    train_dataset = TimeSeriesDataset()
    test_dataset = TimeSeriesDataset()

    if multivariate:
        train_dataset = TimeSeriesDataset(
            [TimeSeries(values=train_df.to_numpy(), labels=train_label.to_numpy())]
        )
        test_dataset = TimeSeriesDataset(
            [TimeSeries(values=test_df.to_numpy(), labels=test_label.to_numpy())]
        )
    else:
        train_dataset = TimeSeriesDataset(
            [
                TimeSeries(
                    values=ts,
                    labels=train_label.to_numpy(),
                )
                for ts in np.hsplit(train_df.to_numpy(), train_df.shape[-1])
            ]
        )
        test_dataset = TimeSeriesDataset(
            [
                TimeSeries(
                    values=ts,
                    labels=test_label.to_numpy(),
                )
                for ts in np.hsplit(test_df.to_numpy(), train_df.shape[-1])
            ]
        )

    print("\n...SWaT dataset loaded succesfully.\n")

    return train_dataset, test_dataset
