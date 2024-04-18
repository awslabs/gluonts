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

from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets


def generate_ercot_dataset(dataset_path: Path, dataset_writer: DatasetWriter):
    url = "https://github.com/ourownstory/neuralprophet-data/raw/main/datasets_raw/energy/ERCOT_load_2004_2021Sept.csv"
    df = pd.read_csv(url)
    # There is only a single missing value per time series - forward fill them
    df.ffill(inplace=True)
    regions = [col for col in df.columns if col not in ["ds", "y"]]

    freq = "1H"
    prediction_length = 24

    start = pd.Period(df["ds"][0], freq=freq)

    test = [
        {
            "start": start,
            "item_id": region,
            "target": df[region].to_numpy(dtype=np.float64),
        }
        for region in regions
    ]

    train = [
        {
            "start": start,
            "item_id": region,
            "target": df[region].to_numpy(dtype=np.float64)[
                :-prediction_length
            ],
        }
        for region in regions
    ]

    metadata = MetaData(freq=freq, prediction_length=prediction_length)
    dataset = TrainDatasets(metadata=metadata, train=train, test=test)
    dataset.save(str(dataset_path), writer=dataset_writer, overwrite=True)
