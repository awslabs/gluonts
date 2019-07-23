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

import json
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel

from gluonts.dataset.common import FileDataset, MetaData
from gluonts.shell import log

from .params import parse_sagemaker_parameters
from .path import ServePaths, TrainPaths
from . import algorithm


def map_value(fn, dct):
    return {key: fn(value) for key, value in dct.items()}


class DataConfig(BaseModel):
    ContentType: Optional[str] = None


# for now we only support train and test
DATASET_NAMES = "train", "test"


class TrainEnv:
    def __init__(self, path: Path = Path("/opt/ml")) -> None:
        self.path = TrainPaths(path)
        self.inputdataconfig = _load_inputdataconfig(self.path.inputdataconfig)
        self.channels = _load_channels(self.path, self.inputdataconfig)
        self.hyperparameters = _load_hyperparameters(
            self.path.hyperparameters, self.channels
        )
        self.current_host = _get_current_host(self.path.resourceconfig)
        self.datasets = _load_datasets(self.hyperparameters, self.channels)

        if "algorithm" in self.channels:
            self.forecaster = algorithm.load(
                self.channels["algorithm"], is_train=True
            )
        else:
            self.forecaster = self.hyperparameters.get("forecaster_name")


class ServeEnv:
    def __init__(self, path: Path = Path("/opt/ml")) -> None:
        self.path = ServePaths(path)

        if (self.path.model / "algorithm").exists():
            # we just need to load the algorithm to run the pre steps again
            algorithm.load(self.path.model / "algorithm")


def _load_inputdataconfig(
    inputdataconfig: Path
) -> Optional[Dict[str, DataConfig]]:
    if inputdataconfig.exists():
        with inputdataconfig.open() as json_file:
            return map_value(DataConfig.parse_obj, json.load(json_file))

    return None


def _load_channels(
    path: TrainPaths, inputdataconfig: Optional[Dict[str, DataConfig]]
) -> Dict[str, Path]:
    """Lists the available channels in `/opt/ml/input/data`.

    Return:
    Dict of channel-names mapping to the corresponding path.

    For DeepAR these are `train` and optionally `test`. For Forecast,
    we also have a `metadata` channel, which just contains some information
    about the dataset in `train` and `test`.

    When running in SageMaker, channels and are listed in
    `/opt/ml/config/inputdataconfig.json`. Thus, if this file is present,
    we take its content to determine which channels are available. To
    support a local development setup, we just list the contents of the
    data folder to get the available channels.
    """
    if inputdataconfig is not None:
        return {name: path.data / name for name in inputdataconfig.keys()}
    else:
        return {channel.name: channel for channel in path.data.iterdir()}


def _load_hyperparameters(path: Path, channels) -> dict:
    with path.open() as json_file:
        hyperparameters = parse_sagemaker_parameters(json.load(json_file))

        for old_freq_name in ["time_freq", "time_granularity", "frequency"]:
            if old_freq_name in hyperparameters:
                hyperparameters["freq"] = hyperparameters[old_freq_name]

        if "metadata" in channels:
            with (channels["metadata"] / "metadata.json").open() as file:
                metadata = MetaData(**json.load(file))
                hyperparameters.update(freq=metadata.freq)

        return hyperparameters


def _get_current_host(resourceconfig: Path) -> str:
    if not resourceconfig.exists():
        return "local"
    else:
        with resourceconfig.open() as json_file:
            config = json.load(json_file)
            return config["current_host"]


def _load_datasets(
    hyperparameters: dict, channels: Dict[str, Path]
) -> Dict[str, FileDataset]:
    freq = hyperparameters["freq"]
    return {
        name: FileDataset(channels[name], freq)
        for name in DATASET_NAMES
        if name in channels
    }
