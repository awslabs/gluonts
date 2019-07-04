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
from typing import Dict

from gluonts.dataset.common import FileDataset, MetaData
from .params import load_sagemaker_hyperparameters
from .path import MLPath


# for now we only support train and test
DATASETS = 'train', 'test'


class SageMakerEnv:
    def __init__(self, path="/opt/ml") -> None:
        ml_path = MLPath(path)
        ml_path.makedirs()

        self.path = ml_path

        self.hyperparameters = load_sagemaker_hyperparameters(
            ml_path.hyperparameters
        )
        self.channels = ml_path.get_channels()
        self.datasets = self._get_datasets()
        self._check_sf2()

    def _get_datasets(self) -> Dict[str, FileDataset]:
        freq = self.hyperparameters["freq"]
        return {
            name: FileDataset(self.channels[name], freq)
            for name in DATASETS
            if name in self.channels
        }

    def _check_sf2(self) -> None:
        if "metadata" in self.channels:
            with (self.channels["metadata"] / "metadata.json").open() as file:
                metadata = MetaData(**json.load(file))
                self.hyperparameters.update(freq=metadata.freq)
