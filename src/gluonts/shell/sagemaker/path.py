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


# Standard library imports
from pathlib import Path


class TrainPaths:
    def __init__(self, base: Path = Path("/opt/ml")) -> None:
        self.base: Path = base.expanduser().resolve()
        self.config: Path = self.base / "input" / "config"
        self.data: Path = self.base / "input" / "data"
        self.model: Path = self.base / "model"
        self.output: Path = self.base / "output"

        self.hyperparameters: Path = self.config / "hyperparameters.json"
        self.inputdataconfig: Path = self.config / "inputdataconfig.json"
        self.resourceconfig: Path = self.config / "resourceconfig.json"

        self.config.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.model.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)


class ServePaths:
    def __init__(self, base: Path = Path("/opt/ml")) -> None:
        self.base: Path = base.expanduser().resolve()
        self.model: Path = self.base / "model"
        self.output: Path = self.base / "output"

        self.model.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
