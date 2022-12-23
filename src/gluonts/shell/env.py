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
import tarfile
from distutils.util import strtobool
from functools import partial
from typing import Dict, Optional, Union

from toolz import valmap

from gluonts.dataset.common import Dataset, FileDataset, MetaData
from gluonts.model import forecast, Predictor

from . import sagemaker


class TrainEnv(sagemaker.TrainEnv):
    def __init__(self, *args, **kwargs):
        sagemaker.TrainEnv.__init__(self, *args, **kwargs)
        self.datasets = self._load()

    def _load(self) -> Dict[str, Union[Dataset, Predictor]]:
        if "metadata" in self.channels:
            path = self.channels.pop("metadata")
            self.hyperparameters["freq"] = MetaData.parse_file(
                path / "metadata.json"
            ).freq

        model: Optional[Predictor] = None
        if "model" in self.channels:
            path = self.channels.pop("model")
            with tarfile.open(path / "model.tar.gz") as targz:
                targz.extractall(path=path)
            model = Predictor.deserialize(path)

        file_dataset = partial(FileDataset, freq=self.hyperparameters["freq"])

        datasets = valmap(file_dataset, self.channels)
        if self._listify_dataset():
            datasets = valmap(list, datasets)
        if model is not None:
            datasets["model"] = model

        return datasets

    def _listify_dataset(self):
        return strtobool(self.hyperparameters.get("listify_dataset", "no"))


class ServeEnv(sagemaker.ServeEnv):
    def __init__(self, *args, **kwargs):
        sagemaker.ServeEnv.__init__(self, *args, **kwargs)

        if self.sagemaker_batch:
            self.batch_config = forecast.Config.parse_raw(
                os.environ["INFERENCE_CONFIG"]
            )
        else:
            self.batch_config = None
