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
from abc import ABC, abstractmethod
from pathlib import Path
from pydoc import locate
from typing import Iterator, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

import pts
from pts.core.serde import dump_json, fqname_for, load_json
from pts.dataset import Dataset, DataEntry, InferenceDataLoader
from pts.transform import Transformation
from .forecast import Forecast
from .forecast_generator import ForecastGenerator, SampleForecastGenerator
from .utils import get_module_forward_input_names

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


class Predictor(ABC):

    __version__: str = pts.__version__

    def __init__(self, prediction_length: int, freq: str) -> None:
        self.prediction_length = prediction_length
        self.freq = freq

    @abstractmethod
    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        pass

    def serialize(self, path: Path) -> None:
        # serialize Predictor type
        with (path / "type.txt").open("w") as fp:
            fp.write(fqname_for(self.__class__))
        with (path / "version.json").open("w") as fp:
            json.dump(
                {"model": self.__version__, "pts": pts.__version__}, fp
            )

    @classmethod
    def deserialize(
            cls, path: Path, device: Optional[torch.device] = None
    ) -> "Predictor":
        """
        Load a serialized predictor from the given path
        Parameters
        ----------
        path
            Path to the serialized files predictor.
        device
            Optional pytorch to be used with the predictor.
            If nothing is passed will use the GPU if available and CPU otherwise.
        """
        # deserialize Predictor type
        with (path / "type.txt").open("r") as fp:
            tpe = locate(fp.readline())

        # ensure that predictor_cls is a subtype of Predictor
        if not issubclass(tpe, Predictor):
            raise IOError(
                f"Class {fqname_for(tpe)} is not "
                f"a subclass of {fqname_for(Predictor)}"
            )
        # call deserialize() for the concrete Predictor type
        return tpe.deserialize(path, device)


class PTSPredictor(Predictor):
    def __init__(
        self,
        prediction_net: nn.Module,
        batch_size: int,
        prediction_length: int,
        freq: str,
        device: torch.device,
        input_transform: Transformation,
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[OutputTransform] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(prediction_length, freq)
        self.input_names = get_module_forward_input_names(prediction_net)
        self.prediction_net = prediction_net
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.forecast_generator = forecast_generator
        self.output_transform = output_transform
        self.device = device
        self.dtype = dtype

    def predict(
        self, dataset: Dataset, num_samples: Optional[int] = None
    ) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            self.input_transform,
            self.batch_size,
            device=self.device,
            dtype=self.dtype,
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

    def serialize(self, path: Path) -> None:

        super().serialize(path)

        # serialize network
        model_name = 'prediction_net'
        with (path / f"{model_name}-network.json").open("w") as fp:
            print(dump_json(self.prediction_net), file=fp)
        torch.save(self.prediction_net.state_dict(), path / "prediction_net")

        # serialize input transformation chain
        with (path / "input_transform.json").open("w") as fp:
            print(dump_json(self.input_transform), file=fp)

        # serialize output transformation chain
        with (path / "output_transform.json").open("w") as fp:
            print(dump_json(self.output_transform), file=fp)

        # serialize all remaining constructor parameters
        with (path / "parameters.json").open("w") as fp:
            parameters = dict(
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                freq=self.freq,
                dtype=self.dtype,
                forecast_generator=self.forecast_generator,
                input_names=self.input_names,
            )
            print(dump_json(parameters), file=fp)

    @classmethod
    def deserialize(
            cls, path: Path, device: Optional[torch.device] = None
    ) -> "PTSPredictor":

        # deserialize constructor parameters
        with (path / "parameters.json").open("r") as fp:
            parameters = load_json(fp.read())

        # deserialize transformation chain
        with (path / "input_transform.json").open("r") as fp:
            transformation = load_json(fp.read())

        # deserialize prediction network
        model_name = 'prediction_net'
        with (path / f"{model_name}-network.json").open("r") as fp:
            prediction_net = load_json(fp.read())
            prediction_net.load_state_dict(torch.load(path / "prediction_net"))

        # input_names is derived from the prediction_net
        if "input_names" in parameters:
            del parameters["input_names"]

        parameters["device"] = device

        return PTSPredictor(
            input_transform=transformation,
            prediction_net=prediction_net,
            **parameters
        )
