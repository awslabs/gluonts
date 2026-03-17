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
from tempfile import TemporaryDirectory
from typing import Optional
from zipfile import ZipFile
from pandas.tseries.frequencies import to_offset

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._tsf_reader import (
    TSFReader,
    frequency_converter,
)
from gluonts.dataset.repository._tsf_datasets import (
    datasets,
    save_datasets,
    save_metadata,
)

from meta.datasets.gluonts import GluonTSDataModule
from meta.datasets.registry import register_data_module


@register_data_module
class DominickDataModule(GluonTSDataModule):
    """
    A data module for the dominick dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _materialize(self, directory: Path) -> None:
        generate_dominick_dataset(
            dataset_path=directory / self.dataset_name,
        )

    @classmethod
    def name(cls) -> str:
        return "dm_dominick"


def generate_dominick_dataset(
    dataset_path: Path,
    prediction_length: Optional[int] = None,
):
    dataset = datasets["dominick"]
    dataset_path.mkdir(exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        with ZipFile(dataset.download(temp_path)) as archive:
            archive.extractall(path=temp_path)

        # only one file is exptected
        reader = TSFReader(temp_path / archive.namelist()[0])
        meta, data = reader.read()

    freq = frequency_converter(meta.frequency)
    if prediction_length is None:
        if hasattr(meta, "forecast_horizon"):
            prediction_length = int(meta.forecast_horizon)
        else:
            prediction_length = default_prediction_length_from_frequency(freq)

    save_metadata(dataset_path, len(data), freq, prediction_length)

    # Impute missing start dates with unix epoch and remove time series whose
    # length is less than or equal to the prediction length
    data = [
        {**d, "start_timestamp": d.get("start_timestamp", "1970-01-01")}
        for d in data
        if len(d[FieldName.TARGET]) > prediction_length
    ]
    save_datasets(dataset_path, data, prediction_length)


def default_prediction_length_from_frequency(freq: str) -> int:
    prediction_length_map = {
        "T": 60,
        "H": 48,
        "D": 30,
        "W": 8,
        "M": 12,
        "Y": 4,
        "W-SUN": 8,  # this frequency is missing in the original gluon-ts code
    }
    try:
        freq = to_offset(freq).name
        return prediction_length_map[freq]
    except KeyError as err:
        raise ValueError(
            f"Cannot obtain default prediction length from frequency `{freq}`."
        ) from err
