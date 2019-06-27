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
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Iterator


class PathsEnvironment(NamedTuple):
    config: Path = Path('/opt/ml/input/config')
    data: Path = Path('/opt/ml/input/data')
    model: Path = Path('/opt/ml/model')
    output: Path = Path('/opt/ml/output')

    @property
    def output_data(self) -> Path:
        return self.output / 'data'

    def makedirs(self) -> None:
        self.config.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.model.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.output_data.mkdir(parents=True, exist_ok=True)
