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

import numpy as np
from mxnet.gluon import HybridBlock
from pandas.tseries.frequencies import to_offset
from pydantic import Field

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
from gluonts.mx.distribution.lds import ParameterBounds
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.itertools import maybe_len
from gluonts.time_feature import (
    TimeFeature,
    norm_freq_str,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    CanonicalInstanceSplitter,
    Chain,
    ExpandDimArray,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    VstackFeatures,
)

from .issm import ISSM, CompositeISSM
from ._network import DeepStatePredictionNetwork, DeepStateTrainingNetwork

SEASON_INDICATORS_FIELD = "seasonal_indicators"


# A dictionary mapping granularity to the period length of the longest season
# one can expect given the granularity of the time series. This is similar to
# the frequency value in the R forecast package:
# https://stats.stackexchange.com/questions/120806/frequency-value-for-seconds-minutes-intervals-data-in-r
# This is useful for setting default values for past/context length for models
# that do not do data augmentation and uses a single training example per time
# series in the dataset.
FREQ_LONGEST_PERIOD_DICT = {
    "M": 12,  # yearly seasonality
    "W": 52,  # yearly seasonality
    "D": 31,  # monthly seasonality
    "B": 22,  # monthly seasonality
    "H": 168,  # weekly seasonality
    "T": 1440,  # daily seasonality
}


def longest_period_from_frequency_str(freq_str: str) -> int:
    offset = to_offset(freq_str)
    return FREQ_LONGEST_PERIOD_DICT[norm_freq_str(offset.name)] // offset.n


@serde.dataclass
class DeepStateEstimator(GluonEstimator):
    """
    Construct a DeepState estimator.

    This implements the deep state space model described in
    [RSG+18]_.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    cardinality
        Number of values of each categorical feature.
        This must be set by default unless ``use_feat_static_cat``
        is set to `False` explicitly (which is NOT recommended).
    add_trend
        Flag to indicate whether to include trend component in the
        state space model
    past_length
        This is the length of the training time series;
        i.e., number of steps to unroll the RNN for before computing
        predictions.
        Set this to (at most) the length of the shortest time series in the
        dataset.
        (default: None, in which case the training length is set such that
        at least
        `num_seasons_to_train` seasons are included in the training.
        See `num_seasons_to_train`)
    num_periods_to_train
        (Used only when `past_length` is not set)
        Number of periods to include in the training time series. (default: 4)
        Here period corresponds to the longest cycle one can expect given
        the granularity of the time series.
        See: https://stats.stackexchange.com/questions/120806/frequency
        -value-for-seconds-minutes-intervals-data-in-r
    trainer
        Trainer object to be used (default: Trainer())
    num_layers
        Number of RNN layers (default: 2)
    num_cells
        Number of RNN cells for each layer (default: 40)
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm')
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference.
        This is a model optimization that does not affect the accuracy (
        default: 100).
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    use_feat_dynamic_real
        Whether to use the ``feat_dynamic_real`` field from the data
        (default: False)
    use_feat_static_cat
        Whether to use the ``feat_static_cat`` field from the data
        (default: True)
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: [min(50, (cat+1)//2) for cat in cardinality])
    scaling
        Whether to automatically scale the target values (default: true)
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    noise_std_bounds
        Lower and upper bounds for the standard deviation of the observation
        noise
    prior_cov_bounds
        Lower and upper bounds for the diagonal of the prior covariance matrix
    innovation_bounds
        Lower and upper bounds for the standard deviation of the observation
        noise
    batch_size
        The size of the batches to be used training and prediction.
    """

    freq: str
    prediction_length: int = Field(..., gt=0)
    cardinality: List[int] = Field(...)
    add_trend: bool = False
    past_length: int = Field(None, gt=0)
    num_periods_to_train: int = 4
    trainer: Trainer = Trainer(
        epochs=100, num_batches_per_epoch=50, hybridize=False
    )
    num_layers: int = Field(2, gt=0)
    num_cells: int = Field(40, gt=0)
    cell_type: str = "lstm"
    num_parallel_samples: int = Field(100, gt=0)
    dropout_rate: float = Field(0.1, ge=0)
    use_feat_dynamic_real: bool = False
    use_feat_static_cat: bool = True
    embedding_dimension: List[int] = Field(None, gt=0)
    issm: ISSM = Field(None)
    scaling: bool = True
    time_features: List[TimeFeature] = Field(None)
    noise_std_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0)
    prior_cov_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0)
    innovation_bounds: ParameterBounds = ParameterBounds(1e-6, 0.01)
    batch_size: int = 32

    def __post_init_post_parse__(self):
        super().__init__(trainer=self.trainer, batch_size=self.batch_size)
        assert not self.use_feat_static_cat or any(
            c > 1 for c in self.cardinality
        ), (
            "Cardinality of at least one static categorical feature must be"
            " larger than 1 if `use_feat_static_cat=True`. But cardinality"
            f" provided is: {self.cardinality}"
        )

        assert all(
            np.isfinite(p.lower) and np.isfinite(p.upper) and p.lower > 0
            for p in [
                self.noise_std_bounds,
                self.prior_cov_bounds,
                self.innovation_bounds,
            ]
        ), (
            "All parameter bounds should be finite, and lower bounds should be"
            " positive"
        )

        if self.past_length is None:
            self.past_length = (
                self.num_periods_to_train
                * longest_period_from_frequency_str(self.freq)
            )

        if self.cardinality is None or not self.use_feat_static_cat:
            self.cardinality = [1]

        if self.embedding_dimension is None:
            self.embedding_dimension = [
                min(50, (cat + 1) // 2) for cat in self.cardinality
            ]

        if self.issm is None:
            self.issm = CompositeISSM.get_from_freq(self.freq, self.add_trend)

        if self.time_features is None:
            self.time_features = time_features_from_frequency_str(self.freq)

    def create_transformation(self) -> Transformation:
        remove_field_names = [
            FieldName.FEAT_DYNAMIC_CAT,
            FieldName.FEAT_STATIC_REAL,
        ]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        empty_list: List[Transformation] = []
        return Chain(
            empty_list
            + [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat
                else []
            )
            + [
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                # gives target the (1, T) layout
                ExpandDimArray(field=FieldName.TARGET, axis=0),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                # Unnormalized seasonal features
                AddTimeFeatures(
                    time_features=self.issm.time_features(),
                    pred_length=self.prediction_length,
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=SEASON_INDICATORS_FIELD,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
            ]
        )

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        return CanonicalInstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            time_series_fields=[
                FieldName.FEAT_TIME,
                SEASON_INDICATORS_FIELD,
                FieldName.OBSERVED_VALUES,
            ],
            allow_target_padding=True,
            instance_length=self.past_length,
            use_prediction_features=(mode != "training"),
            prediction_length=self.prediction_length,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(DeepStateTrainingNetwork)
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
        input_names = get_hybrid_forward_input_names(DeepStateTrainingNetwork)
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> DeepStateTrainingNetwork:
        return DeepStateTrainingNetwork(
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            past_length=self.past_length,
            prediction_length=self.prediction_length,
            issm=self.issm,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            scaling=self.scaling,
            noise_std_bounds=self.noise_std_bounds,
            prior_cov_bounds=self.prior_cov_bounds,
            innovation_bounds=self.innovation_bounds,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = DeepStatePredictionNetwork(
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            past_length=self.past_length,
            prediction_length=self.prediction_length,
            issm=self.issm,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
            noise_std_bounds=self.noise_std_bounds,
            prior_cov_bounds=self.prior_cov_bounds,
            innovation_bounds=self.innovation_bounds,
            params=trained_network.collect_params(),
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
