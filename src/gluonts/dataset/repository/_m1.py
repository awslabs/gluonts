import json
import os
from pathlib import Path
from typing import NamedTuple, List, Dict
from urllib import request
from zipfile import ZipFile

from gluonts.dataset.repository._util import metadata, to_dict, save_to_file

from ._tsf_reader import TSFReader

ROOT = "https://zenodo.org/record"

dataset_info = {
    "m1_yearly": {"file": "m1_yearly_dataset.zip", "record": "4656193"},
    "m1_quarterly": {"file": "m1_quarterly_dataset.zip", "record": "4656154"},
    "m1_monthly": {"file": "m1_monthly_dataset.zip", "record": "4656159"},
}


def download_dataset(description: str, path: Path):
    request.urlretrieve(
        f"{ROOT}/{description['record']}/files/" f"{description['file']}",
        path / description["file"],
        )


def frequency_converter(freq: str):
    split = freq.split("_")
    if len(split) == 1:
        return freq[0].upper()
    elif len(split) == 2 and split[0].isnumeric():
        return f"{split[0]}{split[1].upper()}"
    else:
        raise Exception("Unknown frequency")


def save_metadata(
        dataset_path: Path, cardinality: int, freq: str, prediction_length: int
):
    with open(dataset_path / "metadata.json", "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=cardinality,
                    freq=freq,
                    prediction_length=prediction_length,
                )
            )
        )


def save_dataset(
        dataset_path: Path, data: List[Dict], prediction_length: int = None
):
    train_file = dataset_path / "data.json"
    save_to_file(
        train_file,
        [
            to_dict(
                target_values=data_entry["target"]
                if not prediction_length
                else data_entry["target"][:-prediction_length],
                start=str(data_entry["start_timestamp"]),
            )
            for data_entry in data
        ],
    )


def clean_up_dataset(dataset_path: Path, file_names: [str]):
    for file in file_names:
        os.remove(dataset_path / file)


def generate_forecasting_dataset(dataset_path: Path, dataset_name: str):
    ds_info = dataset_info[dataset_name]
    os.makedirs(dataset_path, exist_ok=True)

    file_path = dataset_path / ds_info["file"]
    download_dataset(ds_info, dataset_path)
    with ZipFile(file_path, "r") as zip:
        file_names = zip.namelist()
        # TODO better exception text
        assert len(file_names) == 1, "Too many files"
        zip.extractall(path=dataset_path)
    reader = TSFReader(dataset_path / file_names[0])
    meta, data = reader.read()

    prediction_length = int(meta.forecast_horizon)
    save_metadata(
        dataset_path,
        len(data),
        frequency_converter(meta.frequency),
        prediction_length,
    )

    save_dataset(dataset_path / "test", data)
    save_dataset(dataset_path / "train", data, prediction_length)

    file_names.append(ds_info["file"])
    clean_up_dataset(dataset_path, file_names)
