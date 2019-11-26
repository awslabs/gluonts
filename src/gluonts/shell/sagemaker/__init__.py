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

import os
import json
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel

from gluonts.dataset.common import FileDataset, MetaData
from gluonts.model.forecast import Config as ForecastConfig
from gluonts.support.util import map_dct_values
from .params import decode_sagemaker_parameters
from .path import ServePaths, TrainPaths


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


class ServeEnv:
    path: ServePaths
    batch_config: Optional[ForecastConfig]

    def __init__(self, path: Path = Path("/opt/ml")) -> None:
        self.path = ServePaths(path)

        batch_transform = os.environ.get("SAGEMAKER_BATCH", "false") == "true"
        if batch_transform:
            self.batch_config = ForecastConfig.parse_raw(
                os.environ["INFERENCE_CONFIG"]
            )
        else:
            self.batch_config = None


def _load_inputdataconfig(
    inputdataconfig: Path,
) -> Optional[Dict[str, DataConfig]]:
    if inputdataconfig.exists():
        with inputdataconfig.open() as json_file:
            return map_dct_values(DataConfig.parse_obj, json.load(json_file))

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
        hyperparameters = decode_sagemaker_parameters(json.load(json_file))

        for old_freq_name in ["time_freq", "time_granularity", "frequency"]:
            if old_freq_name in hyperparameters:
                hyperparameters["freq"] = hyperparameters.pop(old_freq_name)

        if "metadata" in channels:
            with (channels["metadata"] / "metadata.json").open() as file:
                metadata = MetaData(**json.load(file))
                hyperparameters.update(freq=metadata.freq)

        assert "freq" in hyperparameters, (
            "The 'freq' key not in the loaded hyperparameters dictionary. "
            "Please set the 'freq' as a hyperparameter or provide a metadata "
            "channel which contains 'freq' information."
        )

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
