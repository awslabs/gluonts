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
from typing import List

from pydantic import Field

from gluonts import transform
from gluonts.core import serde
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.env import env
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import get_hybrid_forward_input_names
from gluonts.itertools import maybe_len
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    SelectFields,
    SetFieldIfNotPresent,
    TestSplitSampler,
    Transformation,
)

from ._network import DeepFactorPredictionNetwork, DeepFactorTrainingNetwork
from .RNNModel import RNNModel


@serde.dataclass
class DeepFactorEstimator(GluonEstimator):
    r"""
    DeepFactorEstimator is an implementation of the 2019 ICML paper "Deep
    Factors for Forecasting" https://arxiv.org/abs/1905.12417.  It uses a
    global RNN model to learn patterns across multiple related time series and
    an arbitrary local model to model the time series on a per time series
    basis.  In the current implementation, the local model is a RNN (DF-RNN).

    Parameters
    ----------
    freq
        Time series frequency.
    prediction_length
        Prediction length.
    num_hidden_global
        Number of units per hidden layer for the global RNN model
        (default: 50).
    num_layers_global
        Number of hidden layers for the global RNN model (default: 1).
    num_factors
        Number of global factors (default: 10).
    num_hidden_local
        Number of units per hidden layer for the local RNN model (default: 5).
    num_layers_local
        Number of hidden layers for the global local model (default: 1).
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm').
    trainer
        Trainer object to be used (default: Trainer()).
    context_length
        Training length (default: None, in which case context_length =
        prediction_length).
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference. This is a model optimization that does not affect the
        accuracy (default: 100).
    cardinality
        List consisting of the number of time series (default: list([1]).
    embedding_dimension
        Dimension of the embeddings for categorical features (the same
        dimension is used for all embeddings, default: 10).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    batch_size
        The size of the batches to be used training and prediction.
    """

    freq: str
    prediction_length: int = Field(..., gt=0)
    num_hidden_global: int = Field(50, gt=0)
    num_layers_global: int = Field(1, gt=0)
    num_factors: int = Field(10, gt=0)
    num_hidden_local: int = Field(5, gt=0)
    num_layers_local: int = Field(1, gt=0)
    cell_type: str = "lstm"
    trainer: Trainer = Trainer()
    context_length: int = Field(None, gt=0)
    num_parallel_samples: int = Field(100, gt=0)
    cardinality: List[int] = Field([1], gt=0)
    embedding_dimension: int = Field(10, gt=0)
    distr_output: DistributionOutput = StudentTOutput()
    batch_size: int = 32

    def __post_init_post_parse__(self):
        super().__init__(trainer=self.trainer, batch_size=self.batch_size)
        self.context_length = (
            self.context_length
            if self.context_length is not None
            else self.prediction_length
        )
        self.embedding_dimensions = [
            self.embedding_dimension for _ in self.cardinality
        ]

        self.global_model = RNNModel(
            mode=self.cell_type,
            num_hidden=self.num_hidden_global,
            num_layers=self.num_layers_global,
            num_output=self.num_factors,
        )

        # TODO: Allow the local model to be defined as an arbitrary local
        # model, e.g. DF-GP and DF-LDS
        self.local_model = RNNModel(
            mode=self.cell_type,
            num_hidden=self.num_hidden_local,
            num_layers=self.num_layers_local,
            num_output=1,
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            trans=[
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def _create_instance_splitter(self, mode: str):
        return transform.InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            time_series_fields=[FieldName.FEAT_TIME],
            past_length=self.context_length,
            future_length=self.prediction_length,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(DeepFactorTrainingNetwork)
        with env._let(max_idle_transforms=maybe_len(data) or 0):
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
        input_names = get_hybrid_forward_input_names(DeepFactorTrainingNetwork)
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> DeepFactorTrainingNetwork:
        return DeepFactorTrainingNetwork(
            embedder=FeatureEmbedder(
                cardinalities=self.cardinality,
                embedding_dims=self.embedding_dimensions,
            ),
            global_model=self.global_model,
            local_model=self.local_model,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: DeepFactorTrainingNetwork,
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_net = DeepFactorPredictionNetwork(
            embedder=trained_network.embedder,
            global_model=trained_network.global_model,
            local_model=trained_network.local_model,
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
