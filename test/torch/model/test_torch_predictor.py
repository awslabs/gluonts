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

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.predictor import Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import TestSplitSampler, InstanceSplitter


class RandomNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
    ) -> None:
        super().__init__()
        assert prediction_length > 0
        assert context_length > 0
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.net = nn.Linear(context_length, prediction_length)
        torch.nn.init.uniform_(self.net.weight, -1.0, 1.0)

    def forward(self, context):
        assert context.shape[-1] == self.context_length
        out = self.net(context)
        return out.unsqueeze(1)


def test_pytorch_predictor_serde():
    context_length = 20
    prediction_length = 5

    transformation = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=TestSplitSampler(),
        past_length=context_length,
        future_length=prediction_length,
    )

    pred_net = RandomNetwork(
        prediction_length=prediction_length, context_length=context_length
    )

    predictor = PyTorchPredictor(
        prediction_length=prediction_length,
        freq="1H",
        input_names=["past_target"],
        prediction_net=pred_net,
        batch_size=16,
        input_transform=transformation,
        device=torch.device("cpu"),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        predictor.serialize(Path(temp_dir))
        predictor_exp = Predictor.deserialize(Path(temp_dir))
    assert predictor == predictor_exp
