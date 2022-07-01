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

import itertools
from functools import partial
from pathlib import Path
from typing import List, Optional, Callable, Iterator, Type

import mxnet as mx
import numpy as np

from gluonts.core.serde import load_json
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    SampleForecastGenerator,
)
from gluonts.mx.batchify import batchify
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.transform import Transformation


class DeepRenewalProcessSampleOutputTransform:
    """
    Convert a deep renewal process sample that is composed of dense interval-
    size samples to a sparse forecast.

    In practice, this often only means taking the first few time steps of each
    sample trajectory and converting them to the sparse (intermittent)
    representation. Converts a (N, S, 2, T) array corresponding to the
    interval-size format to a (N, S, T) array.
    """

    def __call__(self, entry: DataEntry, output: np.ndarray) -> np.ndarray:
        ia_times, sizes = output[:, :, 0], output[:, :, 1]
        batch_size, num_samples, max_time = ia_times.shape
        max_time = ia_times.shape[-1]
        out = np.zeros_like(ia_times)

        times = np.cumsum(ia_times, axis=-1) - 1

        for i, j in itertools.product(range(batch_size), range(num_samples)):
            t, s = times[i, j], sizes[i, j]
            ix = t[t < max_time].astype(int).tolist()
            if len(ix) > 0:
                out[i, j, ix] = s[: len(ix)]

        return out


class DeepRenewalProcessPredictor(RepresentableBlockPredictor):
    BlockType = mx.gluon.Block

    def __init__(
        self,
        prediction_net: BlockType,
        batch_size: int,
        prediction_length: int,
        ctx: mx.Context,
        input_transform: Transformation,
        input_names: Optional[List[str]] = None,
        lead_time: int = 0,
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[
            Callable[[DataEntry, np.ndarray], np.ndarray]
        ] = DeepRenewalProcessSampleOutputTransform(),
        dtype: Type = np.float32,
    ) -> None:
        super().__init__(
            prediction_net=prediction_net,
            batch_size=batch_size,
            prediction_length=prediction_length,
            ctx=ctx,
            input_transform=input_transform,
            lead_time=lead_time,
            forecast_generator=forecast_generator,
            output_transform=output_transform,
            dtype=dtype,
        )
        if input_names is not None:
            self.input_names = input_names

    def predict(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> Iterator[Forecast]:
        stack_fn = partial(
            batchify,
            ctx=self.ctx,
            dtype=self.dtype,
            variable_length=True,
            is_right_pad=False,
        )
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=stack_fn,
        )
        with mx.Context(self.ctx):
            yield from self.forecast_generator(
                inference_data_loader=inference_data_loader,
                prediction_net=self.prediction_net,
                input_names=self.input_names,
                output_transform=self.output_transform,
                num_samples=num_samples,
            )

    @classmethod
    def deserialize(
        cls, path: Path, ctx: Optional[mx.Context] = None
    ) -> "DeepRenewalProcessPredictor":
        repr_predictor = super().deserialize(path, ctx)
        ctx = repr_predictor.ctx

        with mx.Context(ctx):
            # deserialize constructor parameters
            with (path / "parameters.json").open("r") as fp:
                parameters = load_json(fp.read())
            parameters["ctx"] = ctx

            return DeepRenewalProcessPredictor(
                input_transform=repr_predictor.input_transform,
                prediction_net=repr_predictor.prediction_net,
                **parameters,
            )
