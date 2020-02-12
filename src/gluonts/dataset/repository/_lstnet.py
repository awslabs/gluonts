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
Here we reuse the datasets used by LSTNet as the processed url of the datasets
are available on GitHub.
"""
import json
import os
from pathlib import Path
from typing import List, NamedTuple, Optional

import pandas as pd

from gluonts.dataset.repository._util import metadata, save_to_file, to_dict
from gluonts.support.pandas import frequency_add


def load_from_pandas(
    df: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    agg_freq: Optional[str] = None,
) -> List[pd.Series]:
    df = df.set_index(time_index)

    pivot_df = df.transpose()
    pivot_df.head()

    timeseries = []
    for row in pivot_df.iterrows():
        ts = pd.Series(row[1].values, index=time_index)
        if agg_freq is not None:
            ts = ts.resample(agg_freq).sum()
        first_valid = ts[ts.notnull()].index[0]
        last_valid = ts[ts.notnull()].index[-1]
        ts = ts[first_valid:last_valid]

        timeseries.append(ts)

    return timeseries


class LstnetDataset(NamedTuple):
    name: str
    url: str
    num_series: int
    num_time_steps: int
    prediction_length: int
    rolling_evaluations: int
    freq: str
    start_date: str
    agg_freq: Optional[str] = None


root = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/"

datasets_info = {
    "exchange_rate": LstnetDataset(
        name="exchange_rate",
        url=root + "exchange_rate/exchange_rate.txt.gz",
        num_series=8,
        num_time_steps=7588,
        prediction_length=30,
        rolling_evaluations=5,
        start_date="1990-01-01",
        freq="1B",
        agg_freq=None,
    ),
    "electricity": LstnetDataset(
        name="electricity",
        url=root + "electricity/electricity.txt.gz",
        # original dataset can be found at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
        # the aggregated ones that is used from LSTNet filters out from the initial 370 series the one with no data
        # in 2011
        num_series=321,
        num_time_steps=26304,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2012-01-01",
        freq="1H",
        agg_freq=None,
    ),
    "traffic": LstnetDataset(
        name="traffic",
        url=root + "traffic/traffic.txt.gz",
        # note there are 963 in the original dataset from https://archive.ics.uci.edu/ml/datasets/PEMS-SF
        # but only 862 in LSTNet
        num_series=862,
        num_time_steps=17544,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2015-01-01",
        freq="H",
        agg_freq=None,
    ),
    "solar-energy": LstnetDataset(
        name="solar-energy",
        url=root + "solar-energy/solar_AL.txt.gz",
        num_series=137,
        num_time_steps=52560,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2006-01-01",
        freq="10min",
        agg_freq="1H",
    ),
}


def generate_lstnet_dataset(dataset_path: Path, dataset_name: str):
    ds_info = datasets_info[dataset_name]

    os.makedirs(dataset_path, exist_ok=True)

    with open(dataset_path / "metadata.json", "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=ds_info.num_series,
                    freq=ds_info.freq,
                    prediction_length=ds_info.prediction_length,
                )
            )
        )

    train_file = dataset_path / "train" / "data.json"
    test_file = dataset_path / "test" / "data.json"

    time_index = pd.date_range(
        start=ds_info.start_date,
        freq=ds_info.freq,
        periods=ds_info.num_time_steps,
    )

    df = pd.read_csv(ds_info.url, header=None)

    assert df.shape == (
        ds_info.num_time_steps,
        ds_info.num_series,
    ), f"expected num_time_steps/num_series {(ds_info.num_time_steps, ds_info.num_series)} but got {df.shape}"

    timeseries = load_from_pandas(
        df=df, time_index=time_index, agg_freq=ds_info.agg_freq
    )

    # the last date seen during training
    ts_index = timeseries[0].index
    training_end = ts_index[int(len(ts_index) * (8 / 10))]

    train_ts = []
    for cat, ts in enumerate(timeseries):
        sliced_ts = ts[:training_end]
        if len(sliced_ts) > 0:
            train_ts.append(
                to_dict(
                    target_values=sliced_ts.values,
                    start=sliced_ts.index[0],
                    cat=[cat],
                    item_id=cat,
                )
            )

    assert len(train_ts) == ds_info.num_series

    save_to_file(train_file, train_ts)

    # time of the first prediction
    prediction_dates = [
        frequency_add(training_end, i * ds_info.prediction_length)
        for i in range(ds_info.rolling_evaluations)
    ]

    test_ts = []
    for prediction_start_date in prediction_dates:
        for cat, ts in enumerate(timeseries):
            # print(prediction_start_date)
            prediction_end_date = frequency_add(
                prediction_start_date, ds_info.prediction_length
            )
            sliced_ts = ts[:prediction_end_date]
            test_ts.append(
                to_dict(
                    target_values=sliced_ts.values,
                    start=sliced_ts.index[0],
                    cat=[cat],
                    item_id=cat,
                )
            )

    assert len(test_ts) == ds_info.num_series * ds_info.rolling_evaluations

    save_to_file(test_file, test_ts)
