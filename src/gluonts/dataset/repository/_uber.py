import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from gluonts.dataset.repository._util import metadata


def generate_uber_dataset(
    dataset_path: Path, uber_freq: str, prediction_length: Optional[int] = None
):
    from gluonts.dataset.repository.datasets import default_dataset_path

    uber_csv_path = default_dataset_path / "uber-raw-data-janjune-15.csv"
    if not os.path.exists(uber_csv_path):
        raise RuntimeError(
            "The uber data is available at "
            "https://github.com/fivethirtyeight/uber-tlc-foil-response/blob/"
            "master/uber-trip-data/uber-raw-data-janjune-15.csv.zip "
            "Please download the file and copy the files to this location: "
            f"{uber_csv_path}"
        )

    subsets = {"daily": "1D", "hourly": "1H"}

    assert (
        uber_freq.lower() in subsets
    ), f"invalid uber_freq='{uber_freq}'. Allowed values: {subsets.keys()}"

    freq_setting = subsets[uber_freq.lower()]
    uber_df = pd.read_csv(
        uber_csv_path,
        header=0,
        usecols=["Pickup_date", "locationID"],
        index_col=0,
    )
    # We divide the raw data according to locationID. Each json line represents
    # a time series of a loacationID. The targets are numbers of pickup-events
    # during a day or an hour.
    time_series_of_locations = list(uber_df.groupby(by="locationID"))
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(dataset_path / "train", exist_ok=True)
    os.makedirs(dataset_path / "test", exist_ok=True)
    train_file = dataset_path / "train" / "data.json"
    test_file = dataset_path / "test" / "data.json"
    with open(train_file, "w") as o_train, open(test_file, "w") as o_test:
        for locationID, df in time_series_of_locations:
            df.sort_index()
            df.index = pd.to_datetime(df.index)

            count_series = df.resample(rule=freq_setting).size()
            start_time = pd.Timestamp(df.index[0]).strftime("%Y-%m-%d %X")
            target = count_series.values.tolist()
            feat_static_cat = [locationID] * len(target)
            format_dict = {
                "start": start_time,
                "target": target,
                "feat_static_cat": feat_static_cat,
            }
            test_json_line = json.dumps(format_dict)
            o_test.write(test_json_line)
            o_test.write("\n")
            format_dict["target"] = format_dict["target"][:-prediction_length]
            format_dict["feat_static_cat"] = format_dict["feat_static_cat"][
                :-prediction_length
            ]
            train_json_line = json.dumps(format_dict)
            o_train.write(train_json_line)
            o_train.write("\n")

    with open(dataset_path / "metadata.json", "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=len(time_series_of_locations),
                    freq=freq_setting[1],
                    prediction_length=prediction_length,
                )
            )
        )
