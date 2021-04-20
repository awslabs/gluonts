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
from typing import List, Callable

from mxnet.gluon import HybridBlock, nn

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify, as_in_context
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.rnn import RNN
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import get_hybrid_forward_input_names
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddTimeFeatures,
    AsNumpyArray,
    InstanceSplitter,
    SetFieldIfNotPresent,
    TestSplitSampler,
    ValidationSplitSampler,
    Transformation,
    SelectFields,
)

from ._network import CanonicalPredictionNetwork, CanonicalTrainingNetwork


class CanonicalEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        model: HybridBlock,
        is_sequential: bool,
        freq: str,
        context_length: int,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        num_parallel_samples: int = 100,
        cardinality: List[int] = list([1]),
        embedding_dimension: int = 10,
        distr_output: DistributionOutput = StudentTOutput(),
        batch_size: int = 32,
    ) -> None:
        super().__init__(trainer=trainer, batch_size=batch_size)

        # TODO: error checking
        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.num_parallel_samples = num_parallel_samples
        self.cardinality = cardinality
        self.embedding_dimensions = [embedding_dimension for _ in cardinality]
        self.model = model
        self.is_sequential = is_sequential

    def create_transformation(self) -> Transformation:
        return (
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1)
            + AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(self.freq),
                pred_length=self.prediction_length,
            )
            + SetFieldIfNotPresent(
                field=FieldName.FEAT_STATIC_CAT, value=[0.0]
            )
            + AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1)
        )

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": ValidationSplitSampler(
                min_future=self.prediction_length
            ),
            "validation": ValidationSplitSampler(
                min_future=self.prediction_length
            ),
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            time_series_fields=[FieldName.FEAT_TIME],
            past_length=self.context_length,
            future_length=self.prediction_length,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(CanonicalTrainingNetwork)
        instance_splitter = self._create_instance_splitter("training")
        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            decode_fn=partial(as_in_context, ctx=self.trainer.ctx),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(CanonicalTrainingNetwork)
        instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> CanonicalTrainingNetwork:
        return CanonicalTrainingNetwork(
            embedder=FeatureEmbedder(
                cardinalities=self.cardinality,
                embedding_dims=self.embedding_dimensions,
            ),
            model=self.model,
            distr_output=self.distr_output,
            is_sequential=self.is_sequential,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: CanonicalTrainingNetwork,
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_net = CanonicalPredictionNetwork(
            embedder=trained_network.embedder,
            model=trained_network.model,
            distr_output=trained_network.distr_output,
            is_sequential=trained_network.is_sequential,
            prediction_len=self.prediction_length,
            num_parallel_samples=self.num_parallel_samples,
            params=trained_network.collect_params(),
        )

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_net,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


class CanonicalRNNEstimator(CanonicalEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        num_layers: int = 1,
        num_cells: int = 50,
        cell_type: str = "lstm",
        num_parallel_samples: int = 100,
        cardinality: List[int] = list([1]),
        embedding_dimension: int = 10,
        distr_output: DistributionOutput = StudentTOutput(),
    ) -> None:
        model = RNN(
            mode=cell_type, num_layers=num_layers, num_hidden=num_cells
        )

        super(CanonicalRNNEstimator, self).__init__(
            model=model,
            is_sequential=True,
            freq=freq,
            context_length=context_length,
            prediction_length=prediction_length,
            trainer=trainer,
            num_parallel_samples=num_parallel_samples,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            distr_output=distr_output,
        )


class MLPForecasterEstimator(CanonicalEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        hidden_dim_sequence=list([50]),
        num_parallel_samples: int = 100,
        cardinality: List[int] = list([1]),
        embedding_dimension: int = 10,
        distr_output: DistributionOutput = StudentTOutput(),
    ) -> None:
        model = nn.HybridSequential()

        for layer, layer_dim in enumerate(hidden_dim_sequence):
            model.add(
                nn.Dense(
                    layer_dim,
                    flatten=False,
                    activation="relu",
                    prefix="mlp_%d_" % layer,
                )
            )

        super(MLPForecasterEstimator, self).__init__(
            model=model,
            is_sequential=False,
            freq=freq,
            context_length=context_length,
            prediction_length=prediction_length,
            trainer=trainer,
            num_parallel_samples=num_parallel_samples,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            distr_output=distr_output,
        )
