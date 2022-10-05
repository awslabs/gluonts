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
from typing import Iterator, List, Optional, cast, Type

import mxnet as mx
import numpy as np

from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import DataBatch, InferenceDataLoader
from gluonts.mx.batchify import batchify
from gluonts.mx.model.predictor import (
    GluonPredictor,
    SymbolBlockPredictor,
)
from gluonts.transform import Transformation

from .forecast import PointProcessSampleForecastBatch


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
        prediction_length: int,
        ctx: mx.Context,
        input_transform: Transformation,
        dtype: Type = np.float32,
    ) -> None:
        super().__init__(
            input_names=input_names,
            prediction_net=prediction_net,
            batch_size=batch_size,
            prediction_length=prediction_length,
            ctx=ctx,
            input_transform=input_transform,
            dtype=dtype,
            lead_time=0,
        )

        # not used by TPP predictor
        self.prediction_length = cast(int, None)

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

    def predict_batches(
        self, dataset: Dataset
    ) -> Iterator[PointProcessSampleForecastBatch]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=partial(
                batchify, ctx=self.ctx, dtype=self.dtype, variable_length=True
            ),
        )

        with mx.Context(self.ctx):
            for batch in inference_data_loader:
                yield self.prediction_net.forecast(batch)  # type: ignore

    def serialize_prediction_net(self, path: Path) -> None:
        raise NotImplementedError()
