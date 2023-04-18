from functools import partial
from typing import Callable, Iterator, List, Optional
from pathlib import Path

import mxnet as mx
import numpy as np


from gluonts.core.component import Type
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.loader import DataBatch, InferenceDataLoader
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.model.deepar._network import DeepARPredictionNetwork
from gluonts.core.serde import load_json
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    SampleForecastGenerator,
)
from gluonts.mx.batchify import stack
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.transform import Transformation
from gluonts.mx.util import import_repr_block
from gluonts.mx.context import get_mxnet_context

LOSS_FUNCTIONS = ["crps_univariate", "nll"]


def batchify_with_dict(
    data: List[dict],
    ctx: Optional[mx.context.Context] = None,
    dtype: Optional[Type] = np.float32,
    variable_length: bool = False,
    is_right_pad: bool = True,
) -> DataBatch:
    return {
        key: stack(
            data=[item[key] for item in data],
            ctx=ctx,
            dtype=dtype,
            variable_length=variable_length,
            is_right_pad=is_right_pad,
        )
        if not isinstance(data[0][key], dict)
        else batchify_with_dict(data=[item[key] for item in data])
        for key in data[0].keys()
    }


class RepresentableBlockPredictorBatchifyWithDict(RepresentableBlockPredictor):
    """
    We need the stack function `batchify_with_dict` in order to pass the features at the aggregated level properly
    during prediction. Gluonts does not allow this without changing the line corresponding to the
    `InferenceDataLoader` in the `predict` function.
    """

    def __init__(
        self,
        prediction_net: mx.gluon.HybridBlock,
        batch_size: int,
        prediction_length: int,
        freq: str,
        ctx: mx.Context,
        input_transform: Transformation,
        lead_time: int = 0,
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[
            Callable[[DataEntry, np.ndarray], np.ndarray]
        ] = None,
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
                batchify_with_dict, ctx=self.ctx, dtype=self.dtype
            ),
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
    ) -> "RepresentableBlockPredictorBatchifyWithDict":
        ctx = ctx if ctx is not None else get_mxnet_context()

        with mx.Context(ctx):
            # deserialize constructor parameters
            with (path / "parameters.json").open("r") as fp:
                parameters = load_json(fp.read())

            # deserialize transformation chain
            with (path / "input_transform.json").open("r") as fp:
                transform = load_json(fp.read())

            # deserialize prediction network
            prediction_net = import_repr_block(path, "prediction_net")

            # input_names is derived from the prediction_net
            if "input_names" in parameters:
                del parameters["input_names"]

            parameters["ctx"] = ctx
            return RepresentableBlockPredictorBatchifyWithDict(
                input_transform=transform,
                prediction_net=prediction_net,
                batch_size=parameters["batch_size"],
                freq=parameters["freq"],
                prediction_length=parameters["prediction_length"],
                ctx=parameters["ctx"],
                dtype=parameters["dtype"],
            )


# Gluonts estimator should expose this function. Currently it has api for creating predictor but not prediction network!
def create_prediction_network(
    estimator: DeepAREstimator,
) -> DeepARPredictionNetwork:
    return DeepARPredictionNetwork(
        num_parallel_samples=estimator.num_parallel_samples,
        num_layers=estimator.num_layers,
        num_cells=estimator.num_cells,
        cell_type=estimator.cell_type,
        history_length=estimator.history_length,
        context_length=estimator.context_length,
        prediction_length=estimator.prediction_length,
        distr_output=estimator.distr_output,
        dropoutcell_type=estimator.dropoutcell_type,
        dropout_rate=estimator.dropout_rate,
        cardinality=estimator.cardinality,
        embedding_dimension=estimator.embedding_dimension,
        lags_seq=estimator.lags_seq,
        scaling=estimator.scaling,
        dtype=estimator.dtype,
        num_imputation_samples=estimator.num_imputation_samples,
        default_scale=estimator.default_scale,
        minimum_scale=estimator.minimum_scale,
        impute_missing_values=estimator.impute_missing_values,
    )
