from typing import Union, Iterable
from pathlib import PosixPath

import numpy as np
import pandas as pd

from ncad.ts import TimeSeries, TimeSeriesDataset

from tqdm import tqdm


def yahoo(
    path: Union[PosixPath, str], benchmark_num: Union[int, Iterable[int]], *args, **kwargs
) -> TimeSeriesDataset:
    """Loads Yahoo Webscope Bechmark dataset.

    Args:
        path : Path to the directory containing the four benchmarks and their corresponding csv files.
        benchmark_num : specifies the benchmark(s) number to be loaded. Any subset of [1,2,3,4].
    """
    path = PosixPath(path).expanduser()
    assert path.is_dir(), f"path {path} does not exist"

    # Verify that all subdirectories exist
    bmk_dirs = [f"A{i}Benchmark" for i in range(1, 5)]
    assert np.all([(path / bmk_dir).is_dir() for bmk_dir in bmk_dirs])

    if isinstance(benchmark_num, int):
        benchmark_num = [benchmark_num]
    assert isinstance(benchmark_num, Iterable)

    assert np.all([1 <= i <= 4 for i in benchmark_num])

    dataset = TimeSeriesDataset()
    labels_field = ["is_anomaly", "is_anomaly", "anomaly", "anomaly"]

    for i in benchmark_num:
        bmk_path = path / f"A{i}Benchmark"
        bmk_files = list(bmk_path.glob("*.csv"))
        for ts_path in tqdm(bmk_files):
            ts_pd = pd.read_csv(ts_path)
            if "value" in ts_pd.columns:
                dataset.append(
                    TimeSeries(
                        values=ts_pd["value"],
                        labels=ts_pd[labels_field[i - 1]],
                    )
                )

    return dataset
