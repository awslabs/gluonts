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
from typing import Callable, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn

from gluonts.core.serde import dump_json, load_json
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    SampleForecastGenerator,
    predict_to_numpy,
)
from gluonts.model.predictor import OutputTransform, Predictor
from gluonts.torch.batchify import batchify
from gluonts.torch.component import equals
from gluonts.transform import Transformation


@predict_to_numpy.register(nn.Module)
def _(prediction_net: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    return prediction_net(*inputs).cpu().numpy()


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
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[OutputTransform] = None,
        lead_time: int = 0,
    ) -> None:
        super().__init__(prediction_length, freq=freq, lead_time=lead_time)
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

    def __eq__(self, that):
        if type(self) != type(that):
            return False

        # TODO: also consider equality of the pipelines
        # if not equals(self.input_transform, that.input_transform):
        #    return False

        return equals(
            self.prediction_net.state_dict(),
            that.prediction_net.state_dict(),
        )

    def serialize(self, path: Path) -> None:
        super().serialize(path)

        # serialize network
        with (path / f"prediction_net.json").open("w") as fp:
            print(dump_json(self.prediction_net), file=fp)
        torch.save(
            self.prediction_net.state_dict(), path / "prediction_net_state"
        )

        # serialize transformation chain
        with (path / "input_transform.json").open("w") as fp:
            print(dump_json(self.input_transform), file=fp)

        # FIXME: also needs to serialize the output_transform

        # serialize all remaining constructor parameters
        with (path / "parameters.json").open("w") as fp:
            parameters = dict(
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                freq=self.freq,
                lead_time=self.lead_time,
                forecast_generator=self.forecast_generator,
                input_names=self.input_names,
            )
            print(dump_json(parameters), file=fp)

    @classmethod
    def deserialize(
        cls, path: Path, device: Optional[torch.device] = None
    ) -> "PyTorchPredictor":
        # deserialize constructor parameters
        with (path / "parameters.json").open("r") as fp:
            parameters = load_json(fp.read())

        # deserialize transformation chain
        with (path / "input_transform.json").open("r") as fp:
            transformation = load_json(fp.read())

        # deserialize network
        with (path / f"prediction_net.json").open("r") as fp:
            prediction_net = load_json(fp.read())
        prediction_net.load_state_dict(
            torch.load(path / "prediction_net_state", map_location=device)
        )

        parameters["device"] = device

        return PyTorchPredictor(
            input_transform=transformation,
            prediction_net=prediction_net,
            **parameters,
        )
