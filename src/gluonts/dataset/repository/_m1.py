import json
import os
from pathlib import Path
from typing import List, Dict
from urllib import request
from zipfile import ZipFile
import copy

from ._tsf_reader import TSFReader
from ._util import metadata, to_dict, save_to_file

from gluonts.gluonts_tqdm import tqdm

ROOT = "https://zenodo.org/record"

dataset_info = {
    "m1_yearly": {"file": "m1_yearly_dataset.zip", "record": "4656193"},
    "m1_quarterly": {"file": "m1_quarterly_dataset.zip", "record": "4656154"},
    "m1_monthly": {"file": "m1_monthly_dataset.zip", "record": "4656159"},
    "traffic_hourly": {
        "file": "traffic_hourly_dataset.zip",
        "record": "4656132",
    },
}


def urllib_retrieve_hook(tqdm):
    """Wraps tqdm instance.
    Don'tqdm forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    # >>> with tqdm(...) as tqdm:
    # ...     reporthook = my_hook(tqdm)
    # ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(block=1, block_size=1, tsize=None):
        """
        block  : int, optional
            Number of blocks transferred so far [default: 1].
        block_size  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            tqdm.total = tsize
        tqdm.update((block - last_b[0]) * block_size)
        last_b[0] = block

    return update_to


def download_dataset(description: str, path: Path):
    file = description["file"]
    with tqdm(
        [],
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=5,
        desc=f"download {file}",
    ) as _tqdm:
        request.urlretrieve(
            f"{ROOT}/{description['record']}/files/{file}",
            path / description["file"],
            reporthook=urllib_retrieve_hook(_tqdm),
        )


def frequency_converter(freq: str):
    parts = freq.split("_")
    if len(parts) == 1:
        return freq[0].upper()
    elif len(parts) == 2 and parts[0].isnumeric():
        return f"{parts[0]}{parts[1].upper()}"
    else:
        raise Exception("Unknown frequency")


def save_metadata(
    dataset_path: Path, cardinality: int, freq: str, prediction_length: int
):
    with open(dataset_path / "metadata.json", "w") as file:
        file.write(
            json.dumps(
                metadata(
                    cardinality=cardinality,
                    freq=freq,
                    prediction_length=prediction_length,
                )
            )
        )


def save_dataset(dataset_path: Path, data: List[Dict]):
    save_to_file(
        dataset_path / "data.json",
        [
            to_dict(
                target_values=data_entry["target"],
                start=str(data_entry["start_timestamp"]),
            )
            for data_entry in data
        ],
    )


def create_train_ds(data: List[dict], prediction_length: int):
    train_ds = copy.deepcopy(data)
    for data_entry in data:
        data_entry["target"] = data_entry["target"][:-prediction_length]
    return train_ds


def clean_up_dataset(dataset_path: Path, file_names: List[str]):
    for file in file_names:
        os.remove(dataset_path / file)


def generate_forecasting_dataset(dataset_path: Path, dataset_name: str):
    ds_info = dataset_info[dataset_name]
    dataset_path.mkdir(exist_ok=True)

    file = ds_info["file"]
    file_path = dataset_path / file
    download_dataset(ds_info, dataset_path)
    with ZipFile(file_path, "r") as zip:
        file_names = zip.namelist()
        # TODO better exception text
        assert len(file_names) == 1, "Too many files"
        for member in tqdm(zip.infolist(), desc=f"Extracting {file}"):
            zip.extract(member, dataset_path)
    reader = TSFReader(dataset_path / file_names[0])
    meta, data = reader.read()

    prediction_length = int(meta.forecast_horizon)
    save_metadata(
        dataset_path,
        len(data),
        frequency_converter(meta.frequency),
        prediction_length,
    )
    train_data = create_train_ds(data, prediction_length)
    save_dataset(dataset_path / "train", train_data)
    save_dataset(dataset_path / "test", data)

    file_names.append(file)
    clean_up_dataset(dataset_path, file_names)
