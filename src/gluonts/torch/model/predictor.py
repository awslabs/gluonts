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
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    SampleForecastGenerator,
    to_numpy,
)
from gluonts.model.predictor import OutputTransform, RepresentablePredictor
from gluonts.torch.batchify import batchify
from gluonts.torch.util import resolve_device
from gluonts.transform import SelectFields, Transformation


@to_numpy.register(torch.Tensor)
def _(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


class PyTorchPredictor(RepresentablePredictor):
    @validated()
    def __init__(
        self,
        input_names: List[str],
        prediction_net: nn.Module,
        batch_size: int,
        prediction_length: int,
        input_transform: Transformation,
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[OutputTransform] = None,
        lead_time: int = 0,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        super().__init__(prediction_length, lead_time=lead_time)
        self.input_names = input_names
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.forecast_generator = forecast_generator
        self.output_transform = output_transform
        self.device = resolve_device(device)
        self.prediction_net = prediction_net.to(self.device)
        self.required_fields = ["forecast_start", "item_id", "info"]

    def to(self, device: Union[str, torch.device]) -> "PyTorchPredictor":
        self.device = resolve_device(device)
        self.prediction_net = self.prediction_net.to(self.device)
        return self

    @property
    def network(self) -> nn.Module:
        return self.prediction_net

    def predict(  # type: ignore
        self, dataset: Dataset, num_samples: Optional[int] = None
    ) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform
            + SelectFields(
                self.input_names + self.required_fields, allow_missing=True
            ),
            batch_size=self.batch_size,
            stack_fn=lambda data: batchify(data, self.device),
        )

        self.prediction_net.eval()

        with torch.no_grad():
            yield from self.forecast_generator(
                inference_data_loader=inference_data_loader,
                prediction_net=self.prediction_net,
                input_names=self.input_names,
                output_transform=self.output_transform,
                num_samples=num_samples,
            )

    def serialize(self, path: Path) -> None:
        super().serialize(path)

        torch.save(
            self.prediction_net.state_dict(), path / "prediction-net-state.pt"
        )

    @classmethod
    def deserialize(  # type: ignore
        cls, path: Path, device: Optional[Union[str, torch.device]] = None
    ) -> "PyTorchPredictor":
        predictor = super().deserialize(path)

        assert isinstance(predictor, cls)

        if device is not None:
            device = resolve_device(device)
            predictor.to(device)

        state_dict = torch.load(
            path / "prediction-net-state.pt",
            map_location=device,
        )
        predictor.prediction_net.load_state_dict(state_dict)

        return predictor
