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
from typing import Dict, Optional, Tuple

from pydantic import BaseModel

from .params import decode_sagemaker_parameters
from .nested_params import decode_nested_parameters


class DataConfig(BaseModel):
    ContentType: Optional[str] = None
    TrainingInputMode: Optional[str] = None
    S3DistributionType: Optional[str] = None
    RecordWrapperType: Optional[str] = None


class InpuDataConfig(BaseModel):
    __root__: Dict[str, DataConfig]

    def __getitem__(self, item):
        return self.__root__[item]

    def channels(self):
        return self.__root__

    def channel_names(self):
        return list(self.__root__.keys())


class TrainPaths:
    def __init__(self, base: Path = Path("/opt/ml")) -> None:
        self.base = base.expanduser().resolve()
        self.config = self.base / "input" / "config"
        self.data = self.base / "input" / "data"
        self.model = self.base / "model"
        self.output = self.base / "output"
        self.failure = self.output / "failure"

        self.hyperparameters = self.config / "hyperparameters.json"
        self.inputdataconfig = self.config / "inputdataconfig.json"
        self.resourceconfig = self.config / "resourceconfig.json"

        self.config.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.model.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)


class TrainEnv:
    def __init__(self, path: Path = Path("/opt/ml")) -> None:
        self.path = TrainPaths(path)
        self.inputdataconfig = self._load_inputdataconfig()
        self.channels = self._load_channels()
        self.current_host = self._get_current_host()

        hyperparameters, env = self._load_hyperparameters()
        self.hyperparameters = hyperparameters
        self.env = env

    def _load_inputdataconfig(self) -> Optional[InpuDataConfig]:
        if self.path.inputdataconfig.exists():
            return InpuDataConfig.parse_file(self.path.inputdataconfig)
        return None

    def _load_channels(self) -> Dict[str, Path]:
        """Lists the available channels in `/opt/ml/input/data`.

        Return:
        Dict of channel-names mapping to the corresponding path.

        When running in SageMaker, channels and are listed in
        `/opt/ml/config/inputdataconfig.json`. Thus, if this file is present,
        we take its content to determine which channels are available. To
        support a local development setup, we just list the contents of the
        data folder to get the available channels.
        """
        if self.inputdataconfig is not None:
            return {
                name: self.path.data / name
                for name in self.inputdataconfig.channel_names()
            }
        else:
            return {
                channel.name: channel for channel in self.path.data.iterdir()
            }

    def _get_current_host(self) -> str:
        if not self.path.resourceconfig.exists():
            return "local"
        else:
            with self.path.resourceconfig.open() as json_file:
                config = json.load(json_file)
                return config["current_host"]

    def _load_hyperparameters(self) -> Tuple[dict, Optional[dict]]:
        with self.path.hyperparameters.open() as json_file:
            raw = json.load(json_file)
            decoded = decode_sagemaker_parameters(raw)

            nested = decode_nested_parameters(decoded)
            hyperparameters = nested.get("", {})
            env = nested.get("env", None)
            return hyperparameters, env
