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

import pandas as pd

from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets
from gluonts.dataset.pandas import PandasDataset


URL = (
    "https://raw.githubusercontent.com/AileenNielsen"
    "/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)


def generate_airpassengers_dataset(
    dataset_path: Path,
    dataset_writer: DatasetWriter,
):
    test_split_index = 36

    df = pd.read_csv(
        URL,
        index_col=0,
        parse_dates=True,
    )

    train = PandasDataset(df[:-test_split_index], target="#Passengers")
    test = PandasDataset(df, target="#Passengers")

    meta = MetaData(freq="1M", prediction_length=12)

    dataset = TrainDatasets(metadata=meta, train=train, test=test)
    dataset.save(str(dataset_path), writer=dataset_writer, overwrite=True)
