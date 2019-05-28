from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import os
import json


def to_dict(cat: int, target_values: np.ndarray):
    # the original dataset did not include time stamps, so we use a mock start date for each time series
    # we use the earliest point available in pandas
    mock_start_dataset = "1750-01-01 00:00:00"
    return {
        "start": mock_start_dataset,
        "target": list(target_values),
        "cat": [cat]
    }


def save_to_file(path: str, data: List[Dict]):
    print(f"saving time-series into {path}")
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))


def metadata(cardinality: int, freq: str, prediction_length: int):
    return {
        "time_granularity": freq,
        "prediction_length": prediction_length,
        "feat_static_cat": [
            {
                "name": "time_series_index",
                "cardinality": str(cardinality)
            }
        ],
    }


def generate_m4_dataset(dataset_path: Path, m4_freq: str, pandas_freq: str, prediction_length: int):
    m4_dataset_url = "https://github.com/M4Competition/M4-methods/raw/master/Dataset"
    train_df = pd.read_csv(f'{m4_dataset_url}/Train/{m4_freq}-train.csv', index_col=0)
    test_df = pd.read_csv(f'{m4_dataset_url}/Test/{m4_freq}-test.csv', index_col=0)

    os.makedirs(dataset_path, exist_ok=True)

    with open(dataset_path / 'metadata.json', 'w') as f:
        f.write(json.dumps(metadata(cardinality=len(train_df), freq=pandas_freq, prediction_length=prediction_length)))

    train_file = dataset_path / "train" / "data.json"
    test_file = dataset_path / "test" / "data.json"

    train_target_values = [ts[~np.isnan(ts)] for ts in train_df.values]
    save_to_file(train_file, [to_dict(cat, target) for cat, target in enumerate(train_target_values)])

    test_target_values = [np.hstack([train_ts, test_ts]) for train_ts, test_ts in
                          zip(train_target_values, test_df.values)]
    save_to_file(test_file, [to_dict(cat, target) for cat, target in enumerate(test_target_values)])
