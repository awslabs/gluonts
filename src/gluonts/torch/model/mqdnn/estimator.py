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

from typing import Iterable, Optional, List, Dict, Any, Type

import numpy as np
import torch
from torch.utils.data import DataLoader

from gluonts.dataset import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.util import IterableDataset
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    RenameFields,
    AddConstFeature,
    SetField,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
    InstanceSampler,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor

from .lightning_module import MQDNNLightningModule
from .module import MQCNNModel, MQRNNModel

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "past_feat_dynamic",
    "future_feat_dynamic",
    "feat_static_real",
    "feat_static_cat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class MQDNNEstimator(PyTorchLightningEstimator):
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        is_iqf: bool = False,
        num_past_feat_dynamic_real: int = 0,
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        cardinalities: Optional[List[int]] = None,
        embedding_dimensions: Optional[List[int]] = None,
        add_time_feature: bool = True,
        add_age_feature: bool = False,
        global_hidden_sizes: List[int] = [30, 30],
        global_activation: torch.nn.Module = torch.nn.ReLU(),
        local_hidden_sizes: List[int] = [20, 20],
        local_activation: torch.nn.Module = torch.nn.ReLU(),
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        dtype: Type = np.float32,
    ):
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
            context_length
            if context_length is not None
            else 4 * self.prediction_length
        )
        self.quantiles = (
            quantiles
            if quantiles is not None
            else [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]
        )
        self.is_iqf = is_iqf
        self.num_past_feat_dynamic_real = num_past_feat_dynamic_real
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_real = num_feat_static_real
        self.cardinalities = cardinalities
        self.embedding_dimensions = embedding_dimensions
        self.add_time_feature = add_time_feature
        self.time_features = time_features_from_frequency_str(self.freq)
        self.add_age_feature = add_age_feature
        self.global_hidden_sizes = global_hidden_sizes
        self.global_activation = global_activation
        self.local_hidden_sizes = local_hidden_sizes
        self.local_activation = local_activation
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )
        self.dtype = dtype

    def create_transformation(self) -> Transformation:
        chain = []
        dynamic_feat_fields = []
        remove_field_names = [
            FieldName.FEAT_DYNAMIC_CAT,
        ]

        if self.num_past_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.cardinalities is None:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)

        chain.extend(
            [
                RemoveFields(field_names=remove_field_names),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
            ]
        )

        if self.add_time_feature:
            chain.append(
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                    dtype=self.dtype,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_TIME)

        if self.add_age_feature:
            chain.append(
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    dtype=self.dtype,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_AGE)

        if self.num_feat_dynamic_real > 0:
            chain.append(
                RenameFields({"dynamic_feat": FieldName.FEAT_DYNAMIC_REAL})
            )
            dynamic_feat_fields.append(FieldName.FEAT_DYNAMIC_REAL)

        if len(dynamic_feat_fields) == 0:
            chain.append(
                AddConstFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_CONST,
                    pred_length=self.prediction_length,
                    # For consistency in case with no dynamic features
                    const=0.0,
                    dtype=self.dtype,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_CONST)

        if len(dynamic_feat_fields) > 1:
            chain.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC,
                    input_fields=dynamic_feat_fields,
                )
            )
        elif len(dynamic_feat_fields) == 1:
            chain.append(
                RenameFields({dynamic_feat_fields[0]: FieldName.FEAT_DYNAMIC})
            )

        if self.cardinalities is None:
            chain.append(
                SetField(
                    output_field=FieldName.FEAT_STATIC_CAT,
                    value=np.array([0], dtype=np.int32),
                )
            )

        if self.num_feat_static_real == 0:
            chain.append(
                SetField(
                    output_field=FieldName.FEAT_STATIC_REAL,
                    value=np.array([0], dtype=self.dtype),
                )
            )

        return Chain(chain)

    def _num_feat_dynamic_real(self):
        return (
            self.num_feat_dynamic_real
            + self.add_age_feature
            + len(self.time_features) * self.add_time_feature
        )

    def _num_feat_static_real(self):
        return max(1, self.num_feat_static_real)

    def _create_instance_splitter(
        self, module: MQDNNLightningModule, mode: str
    ):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_DYNAMIC,
                FieldName.OBSERVED_VALUES,
            ],
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: MQDNNLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "training"
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
        self,
        data: Dataset,
        module: MQDNNLightningModule,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "validation"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: MQDNNLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            forecast_generator=QuantileForecastGenerator(
                quantiles=self.quantiles
            ),
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )


class MQCNNEstimator(MQDNNEstimator):
    def __init__(
        self,
        encoder_channels: List[int] = [30, 30, 30],
        encoder_dilations: List[int] = [1, 3, 9],
        encoder_kernel_sizes: List[int] = [7, 3, 3],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder_channels = encoder_channels
        self.encoder_dilations = encoder_dilations
        self.encoder_kernel_sizes = encoder_kernel_sizes

    def create_lightning_module(self) -> MQDNNLightningModule:
        model = MQCNNModel(
            prediction_length=self.prediction_length,
            num_output_quantiles=len(self.quantiles),
            num_feat_dynamic_real=self._num_feat_dynamic_real(),
            num_feat_static_real=self._num_feat_static_real(),
            cardinalities=self.cardinalities or [1],
            embedding_dimensions=self.embedding_dimensions if self.cardinalities else [1],
            encoder_channels=self.encoder_channels,
            encoder_dilations=self.encoder_dilations,
            encoder_kernel_sizes=self.encoder_kernel_sizes,
            decoder_latent_length=self.prediction_length,
            global_hidden_sizes=self.global_hidden_sizes,
            global_activation=self.global_activation,
            local_hidden_sizes=self.local_hidden_sizes,
            local_activation=self.local_activation,
            is_iqf=self.is_iqf,
        )

        return MQDNNLightningModule(
            model=model,
            quantiles=self.quantiles,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class MQRNNEstimator(MQDNNEstimator):
    def __init__(
        self,
        num_layers: int=1,
        encoder_hidden_size: int=50,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.encoder_hidden_size = encoder_hidden_size

    def create_lightning_module(self) -> MQDNNLightningModule:
        model = MQRNNModel(
            prediction_length=self.prediction_length,
            num_output_quantiles=len(self.quantiles),
            num_feat_dynamic_real=self._num_feat_dynamic_real(),
            num_feat_static_real=self._num_feat_static_real(),
            cardinalities=self.cardinalities or [1],
            embedding_dimensions=self.embedding_dimensions if self.cardinalities else [1],
            num_layers=self.num_layers,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_latent_length=self.prediction_length,
            global_hidden_sizes=self.global_hidden_sizes,
            global_activation=self.global_activation,
            local_hidden_sizes=self.local_hidden_sizes,
            local_activation=self.local_activation,
            is_iqf=self.is_iqf,
        )

        return MQDNNLightningModule(
            model=model,
            quantiles=self.quantiles,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
