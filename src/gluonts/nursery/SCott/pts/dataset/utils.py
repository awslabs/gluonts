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


import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rapidjson as json

from .common import TrainDatasets, MetaData
from .file_dataset import FileDataset


def frequency_add(ts: pd.Timestamp, amount: int) -> pd.Timestamp:
    return ts + ts.freq * amount


def forecast_start(entry):
    return frequency_add(entry["start"], len(entry["target"]))


def to_pandas(instance: dict, freq: str = None) -> pd.Series:
    """
    Transform a dictionary into a pandas.Series object, using its
    "start" and "target" fields.

    Parameters
    ----------
    instance
        Dictionary containing the time series data.
    freq
        Frequency to use in the pandas.Series index.

    Returns
    -------
    pandas.Series
        Pandas time series object.
    """
    target = instance["target"]
    start = instance["start"]
    if not freq:
        freq = start.freqstr
    index = pd.date_range(start=start, periods=len(target), freq=freq)
    return pd.Series(target, index=index)


def load_datasets(metadata, train, test, shuffle: bool = False) -> TrainDatasets:
    """
    Loads a dataset given metadata, train and test path.
    Parameters
    ----------
    metadata
        Path to the metadata file
    train
        Path to the training dataset files.
    test
        Path to the test dataset files.
    shuffle
        Return shuffled train data.
    Returns
    -------
    TrainDatasets
        An object collecting metadata, training data, test data.
    """
    meta = MetaData.parse_file(metadata)
    train_ds = FileDataset(train, meta.freq, shuffle=shuffle)
    test_ds = FileDataset(test, meta.freq) if test else None

    return TrainDatasets(metadata=meta, train=train_ds, test=test_ds)


def save_datasets(dataset: TrainDatasets, path_str: str, overwrite=True) -> None:
    """
    Saves an TrainDatasets object to a JSON Lines file.

    Parameters
    ----------
    dataset
        The training datasets.
    path_str
        Where to save the dataset.
    overwrite
        Whether to delete previous version in this folder.
    """
    path = Path(path_str)

    if overwrite:
        shutil.rmtree(path, ignore_errors=True)

    def dump_line(f, line):
        f.write(json.dumps(line).encode("utf-8"))
        f.write("\n".encode("utf-8"))

    (path / "metadata").mkdir(parents=True)
    with open(path / "metadata/metadata.json", "wb") as f:
        dump_line(f, dataset.metadata.dict())

    (path / "train").mkdir(parents=True)
    with open(path / "train/data.json", "wb") as f:
        for entry in dataset.train:
            dump_line(f, serialize_data_entry(entry))

    if dataset.test is not None:
        (path / "test").mkdir(parents=True)
        with open(path / "test/data.json", "wb") as f:
            for entry in dataset.test:
                dump_line(f, serialize_data_entry(entry))


def serialize_data_entry(data):
    """
    Encode the numpy values in the a DataEntry dictionary into lists so the
    dictionary can be JSON serialized.

    Parameters
    ----------
    data
        The dictionary to be transformed.

    Returns
    -------
    Dict
        The transformed dictionary, where all fields where transformed into
        strings.
    """

    def serialize_field(field):
        if isinstance(field, np.ndarray):
            # circumvent https://github.com/micropython/micropython/issues/3511
            nan_ix = np.isnan(field)
            field = field.astype(np.object_)
            field[nan_ix] = "NaN"
            return field.tolist()
        return str(field)

    return {k: serialize_field(v) for k, v in data.items() if v is not None}
