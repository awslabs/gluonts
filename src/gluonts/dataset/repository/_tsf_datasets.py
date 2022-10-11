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
from typing import Dict, List, NamedTuple, Optional
from urllib import request
from zipfile import ZipFile

from pandas.tseries.frequencies import to_offset

from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets
from gluonts.dataset.field_names import FieldName
from gluonts.gluonts_tqdm import tqdm

from ._tsf_reader import TSFReader, frequency_converter
from ._util import metadata, request_retrieve_hook


class Dataset(NamedTuple):
    file_name: str
    record: str
    ROOT: str = "https://zenodo.org/record"

    @property
    def url(self):
        return f"{self.ROOT}/{self.record}/files/{self.file_name}"

    def download(self, path: Path):
        file_path = path / self.file_name
        with tqdm(
            [],
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=5,
            desc=f"Download {self.file_name}:",
        ) as _tqdm:
            request.urlretrieve(
                self.url,
                filename=file_path,
                reporthook=request_retrieve_hook(_tqdm),
            )
        return file_path


datasets = {
    "kaggle_web_traffic_with_missing": Dataset(
        file_name="kaggle_web_traffic_dataset_with_missing_values.zip",
        record="4656080",
    ),
    "kaggle_web_traffic_without_missing": Dataset(
        file_name="kaggle_web_traffic_dataset_without_missing_values.zip",
        record="4656075",
    ),
    "kaggle_web_traffic_weekly": Dataset(
        file_name="kaggle_web_traffic_weekly_dataset.zip",
        record="4656664",
    ),
    "m1_yearly": Dataset(file_name="m1_yearly_dataset.zip", record="4656193"),
    "m1_quarterly": Dataset(
        file_name="m1_quarterly_dataset.zip", record="4656154"
    ),
    "m1_monthly": Dataset(
        file_name="m1_monthly_dataset.zip", record="4656159"
    ),
    "nn5_daily_with_missing": Dataset(
        file_name="nn5_daily_dataset_with_missing_values.zip",
        record="4656110",
    ),
    "nn5_daily_without_missing": Dataset(
        file_name="nn5_daily_dataset_without_missing_values.zip",
        record="4656117",
    ),
    "nn5_weekly": Dataset(
        file_name="nn5_weekly_dataset.zip", record="4656125"
    ),
    "tourism_monthly": Dataset(
        file_name="tourism_monthly_dataset.zip",
        record="4656096",
    ),
    "tourism_quarterly": Dataset(
        file_name="tourism_quarterly_dataset.zip",
        record="4656093",
    ),
    "tourism_yearly": Dataset(
        file_name="tourism_yearly_dataset.zip",
        record="4656103",
    ),
    "cif_2016": Dataset(
        file_name="cif_2016_dataset.zip",
        record="4656042",
    ),
    "london_smart_meters_without_missing": Dataset(
        file_name="london_smart_meters_dataset_without_missing_values.zip",
        record="4656091",
    ),
    "wind_farms_without_missing": Dataset(
        file_name="wind_farms_minutely_dataset_without_missing_values.zip",
        record="4654858",
    ),
    "car_parts_without_missing": Dataset(
        file_name="car_parts_dataset_without_missing_values.zip",
        record="4656021",
    ),
    "dominick": Dataset(
        file_name="dominick_dataset.zip",
        record="4654802",
    ),
    "fred_md": Dataset(
        file_name="fred_md_dataset.zip",
        record="4654833",
    ),
    "pedestrian_counts": Dataset(
        file_name="pedestrian_counts_dataset.zip",
        record="4656626",
    ),
    "hospital": Dataset(
        file_name="hospital_dataset.zip",
        record="4656014",
    ),
    "covid_deaths": Dataset(
        file_name="covid_deaths_dataset.zip",
        record="4656009",
    ),
    "kdd_cup_2018_without_missing": Dataset(
        file_name="kdd_cup_2018_dataset_without_missing_values.zip",
        record="4656756",
    ),
    "weather": Dataset(
        file_name="weather_dataset.zip",
        record="4654822",
    ),
}


def convert_data(
    data: List[Dict],
    train_offset: int,
    default_start_timestamp: Optional[str] = None,
):
    train_data = []
    test_data = []
    for i, data_entry in tqdm(
        enumerate(data), total=len(data), desc="creating json files"
    ):
        # Convert the data to a GluonTS dataset...
        # - `default_start_timestamp` is required for some datasets which
        #   are not listed here since some datasets do not define start
        #   timestamps
        # - `item_id` is added for all datasets ... many datasets provide
        #   the "series_name"
        test_data.append(
            {
                "target": data_entry["target"],
                "start": str(
                    data_entry.get("start_timestamp", default_start_timestamp)
                ),
                "item_id": data_entry.get("series_name", i),
            }
        )

        train_data.append(
            {
                "target": data_entry["target"][:-train_offset],
                "start": str(
                    data_entry.get("start_timestamp", default_start_timestamp)
                ),
                "item_id": data_entry.get("series_name", i),
            }
        )

    return train_data, test_data


def generate_forecasting_dataset(
    dataset_path: Path,
    dataset_name: str,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
):
    dataset = datasets[dataset_name]
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

    # Impute missing start dates with unix epoch and remove time series whose
    # length is less than or equal to the prediction length
    data = [
        {**d, "start_timestamp": d.get("start_timestamp", "1970-01-01")}
        for d in data
        if len(d[FieldName.TARGET]) > prediction_length
    ]
    train_data, test_data = convert_data(data, prediction_length)

    meta = MetaData(
        **metadata(
            cardinality=len(data),
            freq=freq,
            prediction_length=prediction_length,
        )
    )

    dataset = TrainDatasets(metadata=meta, train=train_data, test=test_data)
    dataset.save(
        path_str=str(dataset_path), writer=dataset_writer, overwrite=True
    )


def default_prediction_length_from_frequency(freq: str) -> int:
    prediction_length_map = {
        "T": 60,
        "H": 48,
        "D": 30,
        "W-SUN": 8,
        "M": 12,
        "Y": 4,
    }
    try:
        freq = to_offset(freq).name
        return prediction_length_map[freq]
    except KeyError as err:
        raise ValueError(
            f"Cannot obtain default prediction length from frequency `{freq}`."
        ) from err
