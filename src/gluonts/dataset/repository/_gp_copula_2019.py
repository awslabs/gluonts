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

# flake8: noqa: E501

"""
Loads the datasets used in Salinas et al.

2019 (https://tinyurl.com/woyhhqy). This wrapper downloads and unpacks them so
they don'thave to be attached as large files in GluonTS master.
"""

import os
import shutil
import tarfile
from pathlib import Path
from typing import NamedTuple, Optional
from urllib import request

from gluonts.dataset.common import FileDataset
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._util import metadata
from gluonts.util import safe_extractall


class GPCopulaDataset(NamedTuple):
    name: str
    url: str
    num_series: int
    prediction_length: int
    freq: str
    rolling_evaluations: int
    max_target_dim: Optional[int] = None


root = "https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/"

datasets_info = {
    "exchange_rate_nips": GPCopulaDataset(
        name="exchange_rate_nips",
        url=root + "exchange_rate_nips.tar.gz",
        num_series=8,
        prediction_length=30,
        freq="B",
        rolling_evaluations=5,
        max_target_dim=None,
    ),
    "electricity_nips": GPCopulaDataset(
        name="electricity_nips",
        url=root + "electricity_nips.tar.gz",
        # original dataset can be found at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
        num_series=370,
        prediction_length=24,
        freq="H",
        rolling_evaluations=7,
        max_target_dim=None,
    ),
    "traffic_nips": GPCopulaDataset(
        name="traffic_nips",
        url=root + "traffic_nips.tar.gz",
        # note there are 963 in the original dataset from https://archive.ics.uci.edu/ml/datasets/PEMS-SF
        num_series=963,
        prediction_length=24,
        freq="H",
        rolling_evaluations=7,
        max_target_dim=None,
    ),
    "solar_nips": GPCopulaDataset(
        name="solar-energy",
        url=root + "solar_nips.tar.gz",
        num_series=137,
        prediction_length=24,
        freq="H",
        rolling_evaluations=7,
        max_target_dim=None,
    ),
    "wiki-rolling_nips": GPCopulaDataset(
        name="wiki-rolling_nips",
        # Direct link required for Github Large File Storage (LFS) file.
        url="https://github.com/awslabs/gluonts/raw/1553651ca1fca63a16e012b8927bd9ce72b8e79e/datasets/wiki-rolling_nips.tar.gz",
        num_series=9535,
        prediction_length=30,
        freq="D",
        rolling_evaluations=5,
        max_target_dim=None,
    ),
    "wiki2000_nips": GPCopulaDataset(
        name="wiki2000_nips",
        # Direct link required for Github Large File Storage (LFS) file.
        url="https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz",
        num_series=2000,
        prediction_length=30,
        freq="D",
        rolling_evaluations=5,
        max_target_dim=None,
    ),
    "taxi_30min": GPCopulaDataset(
        name="taxi_30min",
        url=root + "taxi_30min.tar.gz",
        num_series=1214,
        prediction_length=24,
        freq="30min",
        rolling_evaluations=56,
        max_target_dim=None,
    ),
}


def generate_gp_copula_dataset(
    dataset_path: Path,
    dataset_name: str,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
):
    ds_info = datasets_info[dataset_name]

    download_dataset(dataset_path.parent, ds_info)

    train_data = get_data(dataset_path / "train", ds_info)
    test_data = get_data(dataset_path / "test", ds_info)

    meta = MetaData(
        **metadata(
            cardinality=ds_info.num_series,
            freq=ds_info.freq,
            prediction_length=prediction_length or ds_info.prediction_length,
        )
    )

    dataset = TrainDatasets(metadata=meta, train=train_data, test=test_data)
    dataset.save(
        path_str=str(dataset_path), writer=dataset_writer, overwrite=True
    )
    clean_up_dataset(dataset_path, ds_info)


def download_dataset(dataset_path: Path, ds_info: GPCopulaDataset):
    request.urlretrieve(ds_info.url, dataset_path / f"{ds_info.name}.tar.gz")

    with tarfile.open(dataset_path / f"{ds_info.name}.tar.gz") as tar:
        safe_extractall(tar, path=dataset_path)


def get_data(dataset_path: Path, ds_info: GPCopulaDataset):
    return [
        {
            "target": data_entry[FieldName.TARGET],
            "start": data_entry[FieldName.START],
            # Handles adding categorical features of rolling
            # evaluation dates
            "feat_static_cat": [cat % ds_info.num_series],
            "item_id": cat,
        }
        for cat, data_entry in enumerate(
            FileDataset(dataset_path, freq=ds_info.freq, pattern="[!._]*")
        )
    ]


def clean_up_dataset(dataset_path: Path, ds_info: GPCopulaDataset):
    os.remove(dataset_path.parent / f"{ds_info.name}.tar.gz")
    shutil.rmtree(dataset_path / "metadata", ignore_errors=True)
