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

from typing import Optional, Iterable, Dict, Any, List

import torch
import lightning.pytorch as pl

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.time_feature import (
    minute_of_hour,
    hour_of_day,
    day_of_month,
    day_of_week,
    day_of_year,
    month_of_year,
    week_of_year,
)
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    InstanceSampler,
)

from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import Output, StudentTOutput

from .lightning_module import TiDELightningModule

PREDICTION_INPUT_NAMES = [
    "feat_static_real",
    "feat_static_cat",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class TiDEEstimator(PyTorchLightningEstimator):
    """
    An estimator training the TiDE model form the paper
    https://arxiv.org/abs/2304.08424 extended for probabilistic forecasting.

    This class is uses the model defined in ``TiDEModel``,
    and wraps it into a ``TiDELightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``prediction_length``).
    feat_proj_hidden_dim
        Size of the feature projection layer (default: 4).
    encoder_hidden_dim
        Size of the dense encoder layer (default: 4).
    decoder_hidden_dim
        Size of the dense decoder layer (default: 4).
    temporal_hidden_dim
        Size of the temporal decoder layer (default: 4).
    distr_hidden_dim
        Size of the distribution projection layer (default: 4).
    num_layers_encoder
        Number of layers in dense encoder (default: 1).
    num_layers_decoder
        Number of layers in dense decoder (default: 1).
    decoder_output_dim
        Output size of dense decoder (default: 4).
    dropout_rate
        Dropout regularization parameter (default: 0.3).
    num_feat_dynamic_proj
        Output size of feature projection layer (default: 2).
    num_feat_dynamic_real
        Number of dynamic real features in the data (default: 0).
    num_feat_static_real
        Number of static real features in the data (default: 0).
    num_feat_static_cat
        Number of static categorical features in the data (default: 0).
    cardinality
        Number of values of each categorical feature.
        This must be set if ``num_feat_static_cat > 0`` (default: None).
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: ``[16 for cat in cardinality]``).
    layer_norm
        Enable layer normalization or not (default: False).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    patience
        Patience parameter for learning rate scheduler (default: 10).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    scaling
        Which scaling method to use to scale the target values (default: mean).
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
        (default: 50).
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
        feat_proj_hidden_dim: Optional[int] = None,
        encoder_hidden_dim: Optional[int] = None,
        decoder_hidden_dim: Optional[int] = None,
        temporal_hidden_dim: Optional[int] = None,
        distr_hidden_dim: Optional[int] = None,
        num_layers_encoder: Optional[int] = None,
        num_layers_decoder: Optional[int] = None,
        decoder_output_dim: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        num_feat_dynamic_proj: Optional[int] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        num_feat_static_cat: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        layer_norm: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
        scaling: Optional[str] = "mean",
        distr_output: Output = StudentTOutput(),
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
        self.context_length = context_length or prediction_length
        self.feat_proj_hidden_dim = feat_proj_hidden_dim or 4
        self.encoder_hidden_dim = encoder_hidden_dim or 4
        self.decoder_hidden_dim = decoder_hidden_dim or 4
        self.temporal_hidden_dim = temporal_hidden_dim or 4
        self.distr_hidden_dim = distr_hidden_dim or 4
        self.num_layers_encoder = num_layers_encoder or 1
        self.num_layers_decoder = num_layers_decoder or 1
        self.decoder_output_dim = decoder_output_dim or 4
        self.dropout_rate = dropout_rate or 0.3

        self.num_feat_dynamic_proj = num_feat_dynamic_proj or 2
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_real = num_feat_static_real
        self.num_feat_static_cat = num_feat_static_cat
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or self.cardinality is None
            else [16 for cat in self.cardinality]
        )

        self.layer_norm = layer_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.distr_output = distr_output
        self.scaling = scaling
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
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.num_feat_static_cat > 0
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=[
                        minute_of_hour,
                        hour_of_day,
                        day_of_month,
                        day_of_week,
                        day_of_year,
                        month_of_year,
                        week_of_year,
                    ],
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                    drop_inputs=False,
                ),
                AsNumpyArray(FieldName.FEAT_TIME, expected_ndim=2),
            ]
        )

    def create_lightning_module(self) -> pl.LightningModule:
        return TiDELightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            model_kwargs={
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "num_feat_dynamic_real": 7 + self.num_feat_dynamic_real,
                "num_feat_dynamic_proj": self.num_feat_dynamic_proj,
                "num_feat_static_real": max(1, self.num_feat_static_real),
                "num_feat_static_cat": max(1, self.num_feat_static_cat),
                "cardinality": self.cardinality,
                "embedding_dimension": self.embedding_dimension,
                "feat_proj_hidden_dim": self.feat_proj_hidden_dim,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "temporal_hidden_dim": self.temporal_hidden_dim,
                "distr_hidden_dim": self.distr_hidden_dim,
                "decoder_output_dim": self.decoder_output_dim,
                "dropout_rate": self.dropout_rate,
                "num_layers_encoder": self.num_layers_encoder,
                "num_layers_decoder": self.num_layers_decoder,
                "layer_norm": self.layer_norm,
                "distr_output": self.distr_output,
                "scaling": self.scaling,
            },
        )

    def _create_instance_splitter(
        self, module: TiDELightningModule, mode: str
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
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: TiDELightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: TiDELightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            forecast_generator=self.distr_output.forecast_generator,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )
