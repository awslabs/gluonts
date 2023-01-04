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

from typing import List, Optional, Iterable, Dict, Any

import torch
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
)
from gluonts.torch.util import (
    IterableDataset,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform.sampler import InstanceSampler

from .module import TemporalFusionTransformerModel
from .lightning_module import TemporalFusionTransformerLightningModule
from .transformation import TFTInstanceSplitter


PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "feat_dynamic_cat",
    "feat_dynamic_real",
    "past_feat_dynamic_cat",
    "past_feat_dynamic_real",
    "past_target",
    "past_observed_values",
    "past_is_pad",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


def _default_feat_args(dims_or_cardinalities: List[int]):
    if dims_or_cardinalities:
        return dims_or_cardinalities
    else:
        return [1]


class TemporalFusionTransformerEstimator(PyTorchLightningEstimator):
    """
    Estimator class to train a Temporal Fusion Transformer model, as described in [SFG17]_.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length).
    quantiles
    num_heads
    hidden_size
    num_feat_static_real
    num_feat_dynamic_real
    num_past_feat_dynamic_real
    cardinalities_static
    embedding_dimension
    time_features
    lr
    dropout_rate
    patience
    batch_size
    num_batches_per_epoch
    trainer_kwargs
    train_sampler
    validation_sampler
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        num_heads: int = 4,
        hidden_dim: int = 32,
        variable_dim: int = 32,
        static_dims: Optional[List[int]] = None,
        dynamic_dims: Optional[List[int]] = None,
        past_dynamic_dims: Optional[List[int]] = None,
        static_cardinalities: Optional[List[int]] = None,
        dynamic_cardinalities: Optional[List[int]] = None,
        past_dynamic_cardinalities: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        lr: float = 1e-3,
        dropout_rate: float = 0.1,
        patience: int = 10,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        # Model architecture
        self.quantiles = quantiles
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.variable_dim = variable_dim

        self.static_dims = static_dims
        self.dynamic_dims = dynamic_dims
        self.past_dynamic_dims = past_dynamic_dims
        self.static_cardinalities = static_cardinalities
        self.dynamic_cardinalities = dynamic_cardinalities
        self.past_dynamic_cardinalities = past_dynamic_cardinalities

        if time_features is None:
            time_features = time_features_from_frequency_str(self.freq)
        self.time_features = time_features

        # Training procedure
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)

        transforms = Chain(
            [RemoveFields(field_names=remove_field_names)]
            + [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
            ]
        )

        # TODO

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        ts_fields = [FieldName.FEAT_DYNAMIC_CAT, FieldName.FEAT_DYNAMIC_REAL]
        past_ts_fields = [
            FieldName.PAST_FEAT_DYNAMIC_CAT,
            FieldName.PAST_FEAT_DYNAMIC_REAL,
        ]

        return TFTInstanceSplitter(
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            "training"
        ) + SelectFields(TRAINING_INPUT_NAMES)
        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self, data: Dataset, **kwargs
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            "validation"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_lightning_module(
        self,
    ) -> TemporalFusionTransformerLightningModule:
        model = TemporalFusionTransformerModel(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_var=self.variable_dim,
            d_hidden=self.hidden_dim,
            num_heads=self.num_heads,
            quantiles=self.quantiles,
            d_past_feat_dynamic_real=_default_feat_args(
                self.past_dynamic_dims
            ),
            c_past_feat_dynamic_cat=_default_feat_args(
                self.past_dynamic_cardinalities
            ),
            d_feat_dynamic_real=_default_feat_args(
                [1] * len(self.time_features) + self.dynamic_dims
            ),
            c_feat_dynamic_cat=_default_feat_args(self.dynamic_cardinalities),
            d_feat_static_real=_default_feat_args(self.static_dims),
            c_feat_static_cat=_default_feat_args(self.static_cardinalities),
            dropout_rate=self.dropout_rate,
        )
        return TemporalFusionTransformerLightningModule(
            model=model,
            lr=self.lr,
            patience=self.patience,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: TemporalFusionTransformerLightningModule,
    ) -> PyTorchPredictor:
        # TODO
        prediction_splitter = self._create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            forecast_generator=QuantileForecastGenerator(
                quantiles=[str(q) for q in self.quantiles]
            ),
        )
