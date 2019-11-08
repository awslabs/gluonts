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

# Standard library imports
from typing import List, Optional

# First-party imports
from gluonts import transform
from gluonts.block.feature import FeatureEmbedder
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.distribution import DistributionOutput, StudentTOutput

from gluonts.model.deep_factor.RNNModel import RNNModel
from gluonts.model.deep_factor._network import (
    DeepFactorTrainingNetwork,
    DeepFactorPredictionNetwork,
)
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.trainer import Trainer
from gluonts.transform import (
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    SetFieldIfNotPresent,
    TestSplitSampler,
    Transformation,
)


# Third-party imports


class DeepFactorEstimator(GluonEstimator):
    r"""
    DeepFactorEstimator is an implementation of the 2019 ICML paper "Deep Factors for Forecasting"
    https://arxiv.org/abs/1905.12417.  It uses a global RNN model to learn patterns across multiple related time series
    and an arbitrary local model to model the time series on a per time series basis.  In the current implementation,
    the local model is a RNN (DF-RNN).

    Parameters
    ----------
    freq
        Time series frequency.
    prediction_length
        Prediction length.
    num_hidden_global
        Number of units per hidden layer for the global RNN model (default: 50).
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
        Training length (default: None, in which case context_length = prediction_length).
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 100).
    cardinality
        List consisting of the number of time series (default: list([1]).
    embedding_dimension
        Dimension of the embeddings for categorical features (the same
        dimension is used for all embeddings, default: 10).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        num_hidden_global: int = 50,
        num_layers_global: int = 1,
        num_factors: int = 10,
        num_hidden_local: int = 5,
        num_layers_local: int = 1,
        cell_type: str = "lstm",
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_parallel_samples: int = 100,
        cardinality: List[int] = list([1]),
        embedding_dimension: int = 10,
        distr_output: DistributionOutput = StudentTOutput(),
    ) -> None:
        super().__init__(trainer=trainer)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert num_layers_global > 0, "The value of `num_layers` should be > 0"
        assert num_hidden_global > 0, "The value of `num_hidden` should be > 0"
        assert num_factors > 0, "The value of `num_factors` should be > 0"
        assert (
            num_hidden_local > 0
        ), "The value of `num_hidden_local` should be > 0"
        assert (
            num_layers_local > 0
        ), "The value of `num_layers_local` should be > 0"
        assert all(
            [c > 0 for c in cardinality]
        ), "Elements of `cardinality` should be > 0"
        assert (
            embedding_dimension > 0
        ), "The value of `embedding_dimension` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.num_parallel_samples = num_parallel_samples
        self.cardinality = cardinality
        self.embedding_dimensions = [embedding_dimension for _ in cardinality]

        self.global_model = RNNModel(
            mode=cell_type,
            num_hidden=num_hidden_global,
            num_layers=num_layers_global,
            num_output=num_factors,
        )

        # TODO: Allow the local model to be defined as an arbitrary local model, e.g. DF-GP and DF-LDS
        self.local_model = RNNModel(
            mode=cell_type,
            num_hidden=num_hidden_local,
            num_layers=num_layers_local,
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
                transform.InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=TestSplitSampler(),
                    time_series_fields=[FieldName.FEAT_TIME],
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                ),
            ]
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
        prediction_net = DeepFactorPredictionNetwork(
            embedder=trained_network.embedder,
            global_model=trained_network.global_model,
            local_model=trained_network.local_model,
            prediction_len=self.prediction_length,
            num_parallel_samples=self.num_parallel_samples,
            params=trained_network.collect_params(),
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_net,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
