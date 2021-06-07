import json
from pathlib import Path
from pprint import pprint
from typing import NamedTuple
from urllib import request
from zipfile import ZipFile

from _tsf_reader import TSFReader

ROOT = "https://zenodo.org//record/"


class ForecastingdataPOrgDataset(NamedTuple):
    name: str
    record: str
    file: str
    # num_series: int
    # prediction_length: int
    freq: str
    # rolling_evaluations: int
    # max_target_dim: Optional[int] = None

    @property
    def url(self):
        return f"{ROOT}/{self.record}/files/{self.file}"


dataset_recipe = {
    "m1_yearly": {"file": "m1_yearly_dataset.zip", "record": "4656193"},
    "m1_quarterly": {"file": "m1_quarterly_dataset.zip", "record": "4656154"},
    "m1_monthly": {"file": "m1_monthly_dataset.zip", "record": "4656159"},
}


def download_dataset(dataset_path: Path, dataset: ForecastingdataPOrgDataset):
    file_path = dataset_path / dataset.file
    request.urlretrieve(dataset.url, file_path)
    with ZipFile(file_path, "r") as zip:
        file_names = zip.namelist()
        # TODO better exception text
        assert len(file_names) == 1, "Too many files"
        zip.extractall(path=dataset_path)
    reader = TSFReader(dataset_path / file_names[0])
    meta, data = reader.read()

    print(meta, data)


ds = ForecastingdataPOrgDataset(
    name="m1_yearly_dataset",
    record="4656193",
    file="m1_yearly_dataset.zip",
    freq="Y",
)


def generate_m1_dataset(dataset_path: Path, m1_freq: str):
    # TODO
    pass


download_dataset(Path("./bla"), ds)
