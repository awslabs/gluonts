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
from typing import Callable, Iterator, List, Optional
from pathlib import Path

import mxnet as mx
import numpy as np


from gluonts.core.component import Type, validated
from gluonts.core.serde import load_json
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.loader import DataBatch, InferenceDataLoader
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    SampleForecastGenerator,
)
from gluonts.mx.batchify import stack
from gluonts.mx.context import get_mxnet_context
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.model.deepar._network import DeepARPredictionNetwork
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import import_repr_block
from gluonts.time_feature import TimeFeature
from gluonts.transform import InstanceSampler, Transformation
from gluonts.transform.feature import MissingValueImputation


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


# Hack to expose history_length to constructor to allow for proper
# serialization
class DeepAREstimatorForCOP(DeepAREstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        dropoutcell_type: str = "ZoneoutCell",
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        use_feat_static_real: bool = False,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        imputation_method: Optional[MissingValueImputation] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        dtype: Type = np.float32,
        alpha: float = 0.0,
        beta: float = 0.0,
        batch_size: int = 32,
        default_scale: Optional[float] = None,
        minimum_scale: float = 1e-10,
        impute_missing_values: bool = False,
        num_imputation_samples: int = 1,
        history_length: Optional[int] = None,
    ) -> None:
        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            context_length=context_length,
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            dropoutcell_type=dropoutcell_type,
            dropout_rate=dropout_rate,
            use_feat_dynamic_real=use_feat_dynamic_real,
            use_feat_static_cat=use_feat_static_cat,
            use_feat_static_real=use_feat_static_real,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            distr_output=distr_output,
            scaling=scaling,
            lags_seq=lags_seq,
            time_features=time_features,
            num_parallel_samples=num_parallel_samples,
            imputation_method=imputation_method,
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            alpha=alpha,
            beta=beta,
            default_scale=default_scale,
            minimum_scale=minimum_scale,
            impute_missing_values=impute_missing_values,
            num_imputation_samples=num_imputation_samples,
            batch_size=batch_size,
            dtype=dtype,
        )
        self.freq = freq
        self.history_length: int = (
            history_length
            if history_length is not None
            else self.history_length
        )
