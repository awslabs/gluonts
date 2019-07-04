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
from typing import Dict

from pydantic import BaseModel


class DataConfig(BaseModel):
    ContentType: str


def get_input_data_config(config: dict) -> Dict[str, DataConfig]:
    return {key: DataConfig.parse_obj(value) for key, value in config.items()}


def load_input_data_config(path: Path) -> Dict[str, DataConfig]:
    with path.open() as json_file:
        config = json.load(json_file)
        return get_input_data_config(config)


class MLPath:
    def __init__(self, base="/opt/ml") -> None:
        self.base: Path = Path(base).expanduser().resolve()
        self.config: Path = self.base / "input/config"
        self.data: Path = self.base / "input/data"
        self.model: Path = self.base / "model"
        self.output: Path = self.base / "output"

        self.hyperparameters: Path = self.config / "hyperparameters.json"
        self.inputdataconfig: Path = self.config / "inputdataconfig.json"

    def makedirs(self) -> None:
        self.config.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.model.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        # (self.output / 'data').mkdir(parents=True, exist_ok=True)

    def get_channels(self) -> Dict[str, Path]:
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
        if self.inputdataconfig.exists():
            return {
                name: Path(self.data / name)
                for name in load_input_data_config(self.inputdataconfig)
            }
        else:
            return {channel.name: channel for channel in self.data.iterdir()}
