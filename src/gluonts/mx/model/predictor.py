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


import logging
from functools import partial
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Type

import mxnet as mx
import numpy as np
from gluonts.core.component import tensor_to_numpy
from gluonts.core.serde import dump_json, load_json
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import DataBatch, InferenceDataLoader
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import ForecastBatch, _unpack
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.component import equals
from gluonts.mx.context import get_mxnet_context
from gluonts.mx.util import (
    export_repr_block,
    export_symb_block,
    get_hybrid_forward_input_names,
    hybrid_block_to_symbol_block,
    import_repr_block,
    import_symb_block,
)
from gluonts.transform import Transformation

OutputTransform = Callable[[dict, np.ndarray], np.ndarray]


class GluonPredictor(Predictor):
    """
    Base predictor type for Gluon-based models.

    Parameters
    ----------
    input_names
        Input tensor names for the graph
    prediction_net
        Network that will be called for prediction
    batch_size
        Number of time series to predict in a single batch
    prediction_length
        Number of time steps to predict
    input_transform
        Input transformation pipeline
    ctx
        MXNet context to use for computation
    forecast_generator
        Class to generate forecasts from network outputs
    """

    BlockType = mx.gluon.Block

    def __init__(
        self,
        input_names: List[str],
        prediction_net: BlockType,
        batch_size: int,
        prediction_length: int,
        ctx: mx.Context,
        input_transform: Transformation,
        forecast_type: Type,
        lead_time: int = 0,
        dtype: Type = np.float32,
        output_transform: Optional[OutputTransform] = None,
    ) -> None:
        super().__init__(
            lead_time=lead_time,
            prediction_length=prediction_length,
        )

        self.input_names = input_names
        self.prediction_net = prediction_net
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.ctx = ctx
        self.dtype = dtype
        self.forecast_type = forecast_type
        self.output_transform = output_transform

    @property
    def network(self) -> BlockType:
        return self.prediction_net

    def hybridize(self, batch: DataBatch) -> None:
        """
        Hybridizes the underlying prediction network.

        Parameters
        ----------
        batch
            A batch of data to use for the required forward pass after the
            `hybridize()` call.
        """
        self.prediction_net.hybridize(active=True)
        self.prediction_net(*[batch[k] for k in self.input_names])

    def as_symbol_block_predictor(
        self,
        batch: Optional[DataBatch] = None,
        dataset: Optional[Dataset] = None,
    ) -> "SymbolBlockPredictor":
        """
        Returns a variant of the current :class:`GluonPredictor` backed
        by a Gluon `SymbolBlock`. If the current predictor is already a
        :class:`SymbolBlockPredictor`, it just returns itself.

        One of batch or datset must be set.

        Parameters
        ----------
        batch
            A batch of data to use for the required forward pass after the
            `hybridize()` call of the underlying network.
        dataset
            Dataset from which a batch is extracted if batch is not set.

        Returns
        -------
        SymbolBlockPredictor
            A predictor derived from the current one backed by a `SymbolBlock`.
        """
        raise NotImplementedError

    def predict(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
    ) -> Iterator[Forecast]:
        for forecast_batch in self.predict_batches(dataset):
            yield from forecast_batch

    def predict_batches(self, dataset: Dataset) -> Iterator[ForecastBatch]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.ctx, dtype=self.dtype),
        )

        with mx.Context(self.ctx):
            for batch in inference_data_loader:
                inputs = [batch[name] for name in self.input_names]

                outputs = self.prediction_net(*inputs)

                if self.output_transform is not None:
                    outputs = self.output_transform(batch, outputs)

                yield self.forecast_type(
                    outputs,
                    start=batch["forecast_start"],
                    item_id=batch.get("item_id"),
                    info=batch.get("info"),
                )

    def __eq__(self, that):
        if type(self) != type(that):
            return False

        if not equals(self.prediction_length, that.prediction_length):
            return False
        if not equals(self.lead_time, that.lead_time):
            return False
        if not equals(self.input_transform, that.input_transform):
            return False

        return equals(
            self.prediction_net.collect_params(),
            that.prediction_net.collect_params(),
        )

    def serialize(self, path: Path) -> None:
        # call Predictor.serialize() in order to serialize the class name
        super().serialize(path)

        # serialize every GluonPredictor-specific parameters
        # serialize the prediction network
        self.serialize_prediction_net(path)

        # serialize transformation chain
        with (path / "input_transform.json").open("w") as fp:
            print(dump_json(self.input_transform), file=fp)

        with (path / "output_transform.json").open("w") as fp:
            print(dump_json(self.output_transform), file=fp)

        with (path / "forecast_type.json").open("w") as fp:
            print(dump_json(self.forecast_type), file=fp)

        # serialize all remaining constructor parameters
        with (path / "parameters.json").open("w") as fp:
            parameters = dict(
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                lead_time=self.lead_time,
                ctx=self.ctx,
                dtype=self.dtype,
                input_names=self.input_names,
            )
            print(dump_json(parameters), file=fp)

    def serialize_prediction_net(self, path: Path) -> None:
        raise NotImplementedError()


class SymbolBlockPredictor(GluonPredictor):
    """
    A predictor which serializes the network structure as an MXNet symbolic
    graph. Should be used for models deployed in production in order to ensure
    forward-compatibility as GluonTS models evolve.

    Used by the training shell if training is invoked with a hyperparameter
    `use_symbol_block_predictor = True`.
    """

    BlockType = mx.gluon.SymbolBlock

    def as_symbol_block_predictor(
        self,
        batch: Optional[DataBatch] = None,
        dataset: Optional[Dataset] = None,
    ) -> "SymbolBlockPredictor":
        return self

    def serialize_prediction_net(self, path: Path) -> None:
        export_symb_block(self.prediction_net, path, "prediction_net")

    @classmethod
    def deserialize(
        cls, path: Path, ctx: Optional[mx.Context] = None
    ) -> "SymbolBlockPredictor":
        ctx = ctx if ctx is not None else get_mxnet_context()

        with mx.Context(ctx):
            # deserialize constructor parameters
            with (path / "parameters.json").open("r") as fp:
                parameters = load_json(fp.read())

            parameters["ctx"] = ctx

            # deserialize transformation chain
            with (path / "input_transform.json").open("r") as fp:
                transform = load_json(fp.read())

            with (path / "output_transform.json").open("r") as fp:
                output_transform = load_json(fp.read())

            with (path / "forecast_type.json").open("r") as fp:
                forecast_type = load_json(fp.read())

            # deserialize prediction network
            num_inputs = len(parameters["input_names"])
            prediction_net = import_symb_block(
                num_inputs, path, "prediction_net"
            )

            return SymbolBlockPredictor(
                prediction_net=prediction_net,
                input_transform=transform,
                output_transform=output_transform,
                forecast_type=forecast_type,
                **parameters,
            )


class RepresentableBlockPredictor(GluonPredictor):
    """
    A predictor which serializes the network structure using the JSON-
    serialization methods located in `gluonts.core.serde`. Use the following
    logic to create a `RepresentableBlockPredictor` from a trained prediction
    network.

    >>> def create_representable_block_predictor(
    ...        prediction_network: mx.gluon.HybridBlock,
    ...        **kwargs
    ... ) -> RepresentableBlockPredictor:
    ...    return RepresentableBlockPredictor(
    ...        prediction_net=prediction_network,
    ...        **kwargs
    ...    )
    """

    BlockType = mx.gluon.HybridBlock

    def __init__(
        self,
        prediction_net: BlockType,
        batch_size: int,
        prediction_length: int,
        ctx: mx.Context,
        input_transform: Transformation,
        forecast_type: Type,
        lead_time: int = 0,
        dtype: Type = np.float32,
        output_transform: Optional[OutputTransform] = None,
    ) -> None:
        super().__init__(
            input_names=get_hybrid_forward_input_names(type(prediction_net)),
            prediction_net=prediction_net,
            batch_size=batch_size,
            prediction_length=prediction_length,
            ctx=ctx,
            input_transform=input_transform,
            lead_time=lead_time,
            dtype=dtype,
            forecast_type=forecast_type,
            output_transform=output_transform,
        )

    def as_symbol_block_predictor(
        self,
        batch: Optional[DataBatch] = None,
        dataset: Optional[Dataset] = None,
    ) -> SymbolBlockPredictor:

        if batch is None:
            data_loader = InferenceDataLoader(
                dataset,
                transform=self.input_transform,
                batch_size=self.batch_size,
                stack_fn=partial(batchify, ctx=self.ctx, dtype=self.dtype),
            )
            batch = next(iter(data_loader))

        with self.ctx:
            symbol_block_net = hybrid_block_to_symbol_block(
                hb=self.prediction_net,
                data_batch=[batch[k] for k in self.input_names],
            )

        return SymbolBlockPredictor(
            input_names=self.input_names,
            prediction_net=symbol_block_net,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.ctx,
            input_transform=self.input_transform,
            lead_time=self.lead_time,
            dtype=self.dtype,
            forecast_type=self.forecast_type,
            output_transform=self.output_transform,
        )

    def serialize(self, path: Path) -> None:
        logging.warning(
            "Serializing RepresentableBlockPredictor instances does not save "
            "the prediction network structure in a backwards-compatible "
            "manner. Be careful not to use this method in production."
        )
        super().serialize(path)

    def serialize_prediction_net(self, path: Path) -> None:
        export_repr_block(self.prediction_net, path, "prediction_net")

    @classmethod
    def deserialize(
        cls, path: Path, ctx: Optional[mx.Context] = None
    ) -> "RepresentableBlockPredictor":
        ctx = ctx if ctx is not None else get_mxnet_context()

        with mx.Context(ctx):
            # deserialize constructor parameters
            with (path / "parameters.json").open("r") as fp:
                parameters = load_json(fp.read())

            # deserialize transformation chain
            with (path / "input_transform.json").open("r") as fp:
                input_transform = load_json(fp.read())

            with (path / "output_transform.json").open("r") as fp:
                output_transform = load_json(fp.read())

            with (path / "forecast_type.json").open("r") as fp:
                forecast_type = load_json(fp.read())

            # deserialize prediction network
            prediction_net = import_repr_block(path, "prediction_net")

            # input_names is derived from the prediction_net
            if "input_names" in parameters:
                del parameters["input_names"]

            parameters["ctx"] = ctx

            return RepresentableBlockPredictor(
                prediction_net=prediction_net,
                input_transform=input_transform,
                output_transform=output_transform,
                forecast_type=forecast_type,
                **parameters,
            )
