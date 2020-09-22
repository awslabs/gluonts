from typing import List

from gluonts.dataset.loader import InferenceDataLoader

from gluonts.transform import Transformation

from gluonts.model.predictor import Predictor

from typing import Iterator, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import SampleForecastGenerator
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.torch.batchify import batchify

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]

from gluonts.model.forecast_generator import predict_to_numpy


@predict_to_numpy.register(nn.Module)
def _(prediction_net: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    return prediction_net(*inputs).data.numpy()


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
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(prediction_length, freq)
        self.input_names = input_names
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

    @classmethod
    def deserialize(
        cls, path: Path, ctx: Optional[mx.Context] = None
    ) -> "Predictor":
        raise NotImplementedError
