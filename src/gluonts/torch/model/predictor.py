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

from typing import List
from typing import Iterator, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from gluonts.dataset.loader import InferenceDataLoader

from gluonts.transform import Transformation

from gluonts.model.predictor import Predictor


from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import SampleForecastGenerator
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.torch.batchify import batchify
from gluonts.model.forecast_generator import predict_to_numpy


@predict_to_numpy.register(nn.Module)
def _(prediction_net: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    return prediction_net(*inputs).data.numpy()


OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


class PyTorchPredictor(Predictor):
    def __init__(
        self,
        input_names: List[str],
        prediction_net: nn.Module,
        batch_size: int,
        prediction_length: int,
        freq: str,
        device: torch.device,
        input_transform: Transformation,
        forecast_generator: SampleForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[OutputTransform] = None,
    ) -> None:
        super().__init__(prediction_length, freq)
        self.input_names = input_names
        self.prediction_net = prediction_net
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.forecast_generator = forecast_generator
        self.output_transform = output_transform
        self.device = device

    def predict(
        self, dataset: Dataset, num_samples: Optional[int] = None
    ) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=lambda data: batchify(data, self.device),
        )

        self.prediction_net.eval()

        with torch.no_grad():
            yield from self.forecast_generator(
                inference_data_loader=inference_data_loader,
                prediction_net=self.prediction_net,
                input_names=self.input_names,
                freq=self.freq,
                output_transform=self.output_transform,
                num_samples=num_samples,
            )
