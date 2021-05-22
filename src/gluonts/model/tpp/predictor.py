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

from functools import partial
from pathlib import Path
from typing import Callable, Iterator, List, Optional, cast

import mxnet as mx
import numpy as np

from gluonts.core.component import DType
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import DataBatch, DataLoader, InferenceDataLoader
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import ForecastGenerator
from gluonts.model.predictor import OutputTransform
from gluonts.mx.batchify import batchify
from gluonts.mx.model.predictor import GluonPredictor, SymbolBlockPredictor
from gluonts.transform import Transformation

from .forecast import PointProcessSampleForecast


class PointProcessForecastGenerator(ForecastGenerator):
    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net: mx.gluon.Block,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs,
    ) -> Iterator[Forecast]:

        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]

            outputs, valid_length = (
                x.asnumpy() for x in prediction_net(*inputs)
            )
            # outputs (num_parallel_samples, batch_size, max_seq_len, 2)
            # valid_length (num_parallel_samples, batch_size)

            # sample until enough point process trajectories are collected
            if num_samples:
                num_collected_samples = outputs.shape[0]
                collected_samples, collected_vls = [outputs], [valid_length]
                while num_collected_samples < num_samples:
                    outputs, valid_length = (
                        x.asnumpy() for x in prediction_net(*inputs)
                    )

                    collected_samples.append(outputs)
                    collected_vls.append(valid_length)

                    num_collected_samples += outputs.shape[0]

                outputs = np.concatenate(collected_samples)[:num_samples]
                valid_length = np.concatenate(collected_vls)[:num_samples]
                # outputs (num_samples, batch_size, max_seq_len, 2)
                # valid_length (num_samples, batch_size)

                assert outputs.shape[0] == num_samples
                assert valid_length.shape[0] == num_samples

            assert outputs.ndim == 4
            assert valid_length.ndim == 2

            batch_size = outputs.shape[1]
            for i in range(batch_size):
                yield PointProcessSampleForecast(
                    outputs[:, i],
                    valid_length=valid_length[:, i],
                    start_date=batch["forecast_start"][i],
                    freq=freq,
                    prediction_interval_length=prediction_net.prediction_interval_length,
                    item_id=batch["item_id"][i]
                    if "item_id" in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                )


class PointProcessGluonPredictor(GluonPredictor):
    """
    Predictor object for marked temporal point process models.

    TPP predictions differ from standard discrete-time models in several
    regards. First, at least for now, only sample forecasts implementing
    PointProcessSampleForecast are available. Similar to TPP Estimator
    objects, the Predictor works with :code:`prediction_interval_length`
    as opposed to :code:`prediction_length`.

    The predictor also accounts for the fact that the prediction network
    outputs a 2-tuple of Tensors, for the samples themselves and their
    `valid_length`.

    Parameters
    ----------
    prediction_interval_length
        The length of the prediction interval
    """

    def __init__(
        self,
        input_names: List[str],
        prediction_net: mx.gluon.Block,
        batch_size: int,
        prediction_interval_length: float,
        freq: str,
        ctx: mx.Context,
        input_transform: Transformation,
        dtype: DType = np.float32,
        forecast_generator: ForecastGenerator = PointProcessForecastGenerator(),
        **kwargs,
    ) -> None:
        super().__init__(
            input_names=input_names,
            prediction_net=prediction_net,
            batch_size=batch_size,
            prediction_length=np.ceil(
                prediction_interval_length
            ),  # for validation only
            freq=freq,
            ctx=ctx,
            input_transform=input_transform,
            output_transform=None,
            dtype=dtype,
            lead_time=0,
            **kwargs,
        )

        # not used by TPP predictor
        self.prediction_length = cast(int, None)

        self.forecast_generator = forecast_generator
        self.prediction_interval_length = prediction_interval_length

    def hybridize(self, batch: DataBatch) -> None:
        raise NotImplementedError(
            "Point process models are currently not hybridizable"
        )

    def as_symbol_block_predictor(
        self,
        batch: Optional[DataBatch] = None,
        dataset: Optional[Dataset] = None,
    ) -> SymbolBlockPredictor:
        raise NotImplementedError(
            "Point process models are currently not hybridizable"
        )

    def predict(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=partial(
                batchify, ctx=self.ctx, dtype=self.dtype, variable_length=True
            ),
        )
        yield from self.forecast_generator(
            inference_data_loader=inference_data_loader,
            prediction_net=self.prediction_net,
            input_names=self.input_names,
            freq=self.freq,
            output_transform=self.output_transform,
            num_samples=num_samples,
        )

    def serialize_prediction_net(self, path: Path) -> None:
        raise NotImplementedError()
