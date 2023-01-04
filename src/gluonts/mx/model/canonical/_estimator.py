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

from dataclasses import field, InitVar
from functools import partial
from typing import List, Type

import numpy as np
from mxnet.gluon import HybridBlock, nn
from pydantic import Field

from gluonts.core import serde
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.rnn import RNN
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import get_hybrid_forward_input_names
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddTimeFeatures,
    AsNumpyArray,
    InstanceSplitter,
    SelectFields,
    SetFieldIfNotPresent,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)

from ._network import CanonicalPredictionNetwork, CanonicalTrainingNetwork


@serde.dataclass
class CanonicalEstimator(GluonEstimator):
    model: HybridBlock = field(init=False)
    is_sequential: bool = field(init=False)
    freq: str
    context_length: int
    prediction_length: int
    lead_time: int = field(default=0, init=False)
    trainer: Trainer = Trainer()
    num_parallel_samples: int = 100
    cardinality: List[int] = field(default_factory=lambda: [1])
    embedding_dimension: InitVar[int] = 10
    distr_output: DistributionOutput = StudentTOutput()
    batch_size: int = Field(32, ge=1)
    dtype: Type = np.float32

    embedding_dimensions: List[int] = field(init=False)

    def __post_init__(self, embedding_dimension) -> None:
        self.embedding_dimensions = [embedding_dimension] * len(
            self.cardinality
        )

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
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


@serde.dataclass
class CanonicalRNNEstimator(CanonicalEstimator):
    num_layers: int = 1
    num_cells: int = 50
    cell_type: str = "lstm"

    def __post_init__(self, embedding_dimension):
        self.is_sequential = False
        self.model = RNN(
            mode=self.cell_type,
            num_layers=self.num_layers,
            num_hidden=self.num_cells,
        )

        super().__post_init__(embedding_dimension)


@serde.dataclass
class MLPForecasterEstimator(CanonicalEstimator):
    hidden_dim_sequence: List[int] = field(default_factory=lambda: [50])
    is_sequential: bool = field(default=False, init=False)

    def __post_init__(self, embedding_dimension):
        self.is_sequential = False
        self.model = nn.HybridSequential()

        super().__post_init__(embedding_dimension)

        for layer, layer_dim in enumerate(self.hidden_dim_sequence):
            self.model.add(
                nn.Dense(
                    layer_dim,
                    flatten=False,
                    activation="relu",
                    prefix="mlp_%d_" % layer,
                )
            )
