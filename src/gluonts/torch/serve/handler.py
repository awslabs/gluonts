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

from pathlib import Path
import torch
from ts.torch_handler.base_handler import BaseHandler
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.predictor import PyTorchPredictor


class GluonTSHandler(BaseHandler):
    def initialize(self, context):
        # Partially copied from BaseHandler::initialize
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available()
            and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        self.predictor = PyTorchPredictor.deserialize(Path(model_dir))
        self.model = self.predictor.prediction_net
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def inference(self, data, *args, **kwargs):
        # FIXME: Get freq from model
        list_dataset = ListDataset(data, freq="1H")
        return list(self.predictor.predict(list_dataset))

    def preprocess(self, data):
        return data

    def postprocess(self, data):
        # FIXME: Return a more complete/configurable response
        return [dict(mean=fcst.mean.tolist()) for fcst in data]
