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

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddConstFeature,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    RemoveFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.transform.split import TFTInstanceSplitter

from .lightning_module import TemporalFusionTransformerLightningModule

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "feat_static_real",
    "feat_static_cat",
    "feat_dynamic_real",
    "feat_dynamic_cat",
    "past_feat_dynamic_real",
    "past_feat_dynamic_cat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class TemporalFusionTransformerEstimator(PyTorchLightningEstimator):
    """
    Estimator class to train a Temporal Fusion Transformer (TFT) model, as described in [LAL+21]_.

    TFT internally performs feature selection when making forecasts. For this
    reason, the dimensions of real-valued features can be grouped together if
    they correspond to the same variable (e.g., treat weather features as a
    one feature and holiday indicators as another feature).

    For example, if the dataset contains key "feat_static_real" with shape
    [batch_size, 3], we can, e.g.,
    - set ``static_dims = [3]`` to treat all three dimensions as a single feature
    - set ``static_dims = [1, 1, 1]`` to treat each dimension as a separate feature
    - set ``static_dims = [2, 1]`` to treat the first two dims as a single feature

    See ``gluonts.torch.model.tft.TemporalFusionTransformerModel.input_shapes``
    for more details on how the model configuration corresponds to the expected
    input shapes.


    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of previous time series values provided as input to the encoder.
        (default: None, in which case context_length = prediction_length).
    quantiles
        List of quantiles that the model will learn to predict.
        Defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_heads
        Number of attention heads in self-attention layer in the decoder.
    hidden_dim
        Size of the LSTM & transformer hidden states.
    variable_dim
        Size of the feature embeddings.
    static_dims
        Sizes of the real-valued static features.
    dynamic_dims
        Sizes of the real-valued dynamic features that are known in the future.
    past_dynamic_dims
        Sizes of the real-valued dynamic features that are only known in the past.
    static_cardinalities
        Cardinalities of the categorical static features.
    dynamic_cardinalities
        Cardinalities of the categorical dynamic features that are known in the future.
    past_dynamic_cardinalities
        Cardinalities of the categorical dynamic features that are ony known in the past.
    time_features
        List of time features, from :py:mod:`gluonts.time_feature`, to use as
        dynamic real features in addition to the provided data (default: None,
        in which case these are automatically determined based on freq).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay (default: ``1e-8``).
    dropout_rate
        Dropout regularization parameter (default: 0.1).
    patience
        Patience parameter for learning rate scheduler.
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch: int = 50,
        Number of batches to be processed in each training epoch (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
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
        weight_decay: float = 1e-8,
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
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantiles = quantiles
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.variable_dim = variable_dim

        self.static_dims = static_dims or []
        self.dynamic_dims = dynamic_dims or []
        self.past_dynamic_dims = past_dynamic_dims or []
        self.static_cardinalities = static_cardinalities or []
        self.dynamic_cardinalities = dynamic_cardinalities or []
        self.past_dynamic_cardinalities = past_dynamic_cardinalities or []

        if time_features is None:
            time_features = time_features_from_frequency_str(self.freq)
        self.time_features = time_features

        # Training procedure
        self.lr = lr
        self.weight_decay = weight_decay
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
        if not self.static_dims:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.dynamic_dims:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if not self.past_dynamic_dims:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if not self.static_cardinalities:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)
        if not self.dynamic_cardinalities:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_CAT)
        if not self.past_dynamic_cardinalities:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_CAT)

        transforms = [
            RemoveFields(field_names=remove_field_names),
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
        ]
        if len(self.time_features) > 0:
            transforms.append(
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                )
            )
        else:
            # Add dummy dynamic feature if no time features are available
            transforms.append(
                AddConstFeature(
                    output_field=FieldName.FEAT_TIME,
                    target_field=FieldName.TARGET,
                    pred_length=self.prediction_length,
                    const=0.0,
                )
            )

        # Provide dummy values if static features are missing
        if not self.static_dims:
            transforms.append(
                SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])
            )
        transforms.append(
            AsNumpyArray(field=FieldName.FEAT_STATIC_REAL, expected_ndim=1)
        )
        if not self.static_cardinalities:
            transforms.append(
                SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])
            )
        transforms.append(
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=np.int64,
            )
        )

        # Concat time features with known dynamic features
        input_fields = [FieldName.FEAT_TIME]
        if self.dynamic_dims:
            input_fields += [FieldName.FEAT_DYNAMIC_REAL]
        transforms.append(
            VstackFeatures(
                input_fields=input_fields,
                output_field=FieldName.FEAT_DYNAMIC_REAL,
            )
        )

        return Chain(transforms)

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        ts_fields = [FieldName.FEAT_DYNAMIC_REAL]
        if self.dynamic_cardinalities:
            ts_fields.append(FieldName.FEAT_DYNAMIC_CAT)
        past_ts_fields = []
        if self.past_dynamic_cardinalities:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_CAT)
        if self.past_dynamic_dims:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)

        return TFTInstanceSplitter(
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

    def input_names(self):
        input_names = list(TRAINING_INPUT_NAMES)

        if not self.dynamic_cardinalities:
            input_names.remove("feat_dynamic_cat")

        if not self.past_dynamic_cardinalities:
            input_names.remove("past_feat_dynamic_cat")

        if not self.past_dynamic_dims:
            input_names.remove("past_feat_dynamic_real")

        return input_names

    def create_training_data_loader(
        self,
        data: Dataset,
        module: TemporalFusionTransformerLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter("training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=self.input_names(),
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: TemporalFusionTransformerLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter("validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=self.input_names(),
            output_type=torch.tensor,
        )

    def create_lightning_module(
        self,
    ) -> TemporalFusionTransformerLightningModule:
        return TemporalFusionTransformerLightningModule(
            lr=self.lr,
            patience=self.patience,
            weight_decay=self.weight_decay,
            model_kwargs={
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "d_var": self.variable_dim,
                "d_hidden": self.hidden_dim,
                "num_heads": self.num_heads,
                "quantiles": self.quantiles,
                "d_past_feat_dynamic_real": self.past_dynamic_dims,
                "c_past_feat_dynamic_cat": self.past_dynamic_cardinalities,
                "d_feat_dynamic_real": [1] * max(len(self.time_features), 1)
                + self.dynamic_dims,
                "c_feat_dynamic_cat": self.dynamic_cardinalities,
                "d_feat_static_real": self.static_dims or [1],
                "c_feat_static_cat": self.static_cardinalities or [1],
                "dropout_rate": self.dropout_rate,
            },
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
