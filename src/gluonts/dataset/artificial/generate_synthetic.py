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

import csv
import json
import os
from typing import List, TextIO

import holidays
import pandas as pd

from gluonts.dataset.artificial._base import (
    ArtificialDataset,
    ComplexSeasonalTimeSeries,
    ConstantDataset,
)
from gluonts.dataset.field_names import FieldName


def write_csv_row(
    time_series: List,
    freq: str,
    csv_file: TextIO,
    is_missing: bool,
    num_missing: int,
) -> None:
    csv_writer = csv.writer(csv_file)
    # convert to right date where MON == 0, ..., SUN == 6
    week_dict = {
        0: "MON",
        1: "TUE",
        2: "WED",
        3: "THU",
        4: "FRI",
        5: "SAT",
        6: "SUN",
    }
    for i in range(len(time_series)):
        data = time_series[i]
        timestamp = pd.Timestamp(data[FieldName.START])
        freq_week_start = freq
        if freq_week_start == "W":
            freq_week_start = f"W-{week_dict[timestamp.weekday()]}"
        timestamp = pd.Timestamp(data[FieldName.START], freq=freq_week_start)
        item_id = int(data[FieldName.ITEM_ID])
        for j, target in enumerate(data[FieldName.TARGET]):
            # Using convention that there are no missing values before the start date
            if is_missing and j != 0 and j % num_missing == 0:
                timestamp += 1
                continue  # Skip every 4th entry
            else:
                timestamp_row = timestamp
                if freq in ["W", "D", "M"]:
                    timestamp_row = timestamp.date()
                row = [item_id, timestamp_row, target]
                # Check if related time series is present
                if FieldName.FEAT_DYNAMIC_REAL in data.keys():
                    for feat_dynamic_real in data[FieldName.FEAT_DYNAMIC_REAL]:
                        row.append(feat_dynamic_real[j])
                csv_writer.writerow(row)
                timestamp += 1


def generate_sf2(
    filename: str, time_series: List, is_missing: bool, num_missing: int
) -> None:
    #  This function generates the test and train json files which will be converted to csv format
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as json_file:
        for ts in time_series:
            if is_missing:
                target = []  # type: List
                # For Forecast don't output feat_static_cat and feat_static_real
                for j, val in enumerate(ts[FieldName.TARGET]):
                    # only add ones that are not missing
                    if j != 0 and j % num_missing == 0:
                        target.append(None)
                    else:
                        target.append(val)
                ts[FieldName.TARGET] = target
            ts.pop(FieldName.FEAT_STATIC_CAT, None)
            ts.pop(FieldName.FEAT_STATIC_REAL, None)
            # Chop features in training set
            if (
                FieldName.FEAT_DYNAMIC_REAL in ts.keys()
                and "train" in filename
            ):
                # TODO: Fix for missing values
                for i, feat_dynamic_real in enumerate(
                    ts[FieldName.FEAT_DYNAMIC_REAL]
                ):
                    ts[FieldName.FEAT_DYNAMIC_REAL][i] = feat_dynamic_real[
                        : len(ts[FieldName.TARGET])
                    ]
            json.dump(ts, json_file)
            json_file.write("\n")


def generate_sf2s_and_csv(
    file_path: str,
    folder_name: str,
    artificial_dataset: ArtificialDataset,
    is_missing: bool = False,
    num_missing: int = 4,
) -> None:
    file_path += f"{folder_name}"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    freq = artificial_dataset.metadata.freq
    train_set = artificial_dataset.train
    generate_sf2(file_path + "train.json", train_set, is_missing, num_missing)
    test_set = artificial_dataset.test
    generate_sf2(file_path + "test.json", test_set, is_missing, num_missing)
    with open(file_path + "input_to_forecast.csv", "w") as csv_file:
        # Test set has training set with the additional values to predict
        write_csv_row(test_set, freq, csv_file, is_missing, num_missing)


if __name__ == "__main__":
    num_timeseries = 1
    file_path = "../../../datasets/synthetic/"
    generate_sf2s_and_csv(file_path, "constant/", ConstantDataset())
    generate_sf2s_and_csv(
        file_path, "constant_missing/", ConstantDataset(), is_missing=True
    )
    generate_sf2s_and_csv(
        file_path, "constant_random/", ConstantDataset(is_random_constant=True)
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_one_ts/",
        ConstantDataset(
            num_timeseries=num_timeseries, is_random_constant=True
        ),
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_diff_scales/",
        ConstantDataset(is_different_scales=True),
    )
    generate_sf2s_and_csv(
        file_path, "constant_noise/", ConstantDataset(is_noise=True)
    )
    generate_sf2s_and_csv(
        file_path, "constant_linear_trend/", ConstantDataset(is_trend=True)
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_linear_trend_noise/",
        ConstantDataset(is_noise=True, is_trend=True),
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_noise_long/",
        ConstantDataset(is_noise=True, is_long=True),
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_noise_short/",
        ConstantDataset(is_noise=True, is_short=True),
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_diff_scales_noise/",
        ConstantDataset(is_noise=True, is_different_scales=True),
    )
    generate_sf2s_and_csv(
        file_path, "constant_zeros_and_nans/", ConstantDataset(is_nan=True)
    )
    generate_sf2s_and_csv(  # Requires is_random_constant to be set to True
        file_path,
        "constant_piecewise/",
        ConstantDataset(is_piecewise=True, is_random_constant=True),
    )
    generate_sf2s_and_csv(
        file_path, "complex_seasonal_noise_scale/", ComplexSeasonalTimeSeries()
    )
    generate_sf2s_and_csv(
        file_path,
        "complex_seasonal_noise/",
        ComplexSeasonalTimeSeries(is_scale=False),
    )
    generate_sf2s_and_csv(
        file_path,
        "complex_seasonal/",
        ComplexSeasonalTimeSeries(is_scale=False, is_noise=False),
    )
    generate_sf2s_and_csv(
        file_path,
        "complex_seasonal_missing/",
        ComplexSeasonalTimeSeries(proportion_missing_values=0.8),
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_missing_middle/",
        ConstantDataset(num_steps=500, num_missing_middle=100),
    )
    generate_sf2s_and_csv(
        file_path,
        "complex_seasonal_random_start_dates_weekly/",
        ComplexSeasonalTimeSeries(
            freq_str="W",
            percentage_unique_timestamps=1,
            is_out_of_bounds_date=True,
        ),
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_promotions/",
        ConstantDataset(
            is_promotions=True,
            freq="M",
            start="2015-11-30",
            num_timeseries=100,
            num_steps=50,
        ),
    )
    generate_sf2s_and_csv(
        file_path,
        "constant_holidays/",
        ConstantDataset(
            start="2017-07-01",
            freq="D",
            holidays=list(holidays.UnitedStates(years=[2017, 2018]).keys()),
            num_steps=365,
        ),
    )
