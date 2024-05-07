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

from typing import Any, Dict, List, Optional, Iterable

import lightning.pytorch as pl
import torch
import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.time_feature import (
    get_seasonality,
    time_features_from_frequency_str,
    TimeFeature,
)
from gluonts.transform import (
    Transformation,
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    SetField,
    RemoveFields,
    QuantizeMeanScaled,
    VstackFeatures,
    Identity,
    TestSplitSampler,
    ValidationSplitSampler,
)
from gluonts.maybe import unwrap_or

from .module import WaveNet
from .lightning_module import WaveNetLightningModule

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_target",
    "past_observed_values",
    "past_time_feat",
    "future_time_feat",
    "scale",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class WaveNetEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        num_bins: int = 1024,
        num_residual_channels: int = 24,
        num_skip_channels: int = 32,
        dilation_depth: Optional[int] = None,
        num_stacks: int = 1,
        temperature: float = 1.0,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: List[int] = [1],
        seasonality: Optional[int] = None,
        embedding_dimension: int = 5,
        use_log_scale_feature: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        num_parallel_samples: int = 100,
        negative_data: bool = False,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        WaveNet estimator that uses the architecture proposed in.

        [Oord et al., 2016] with quantized targets. The model is trained
        using the cross-entropy loss.

        [Oord et al., 2016] "Wavenet: A generative model for raw audio."
        arXiv preprint arXiv:1609.03499 (2016).

        Parameters
        ----------
        freq
            Frequency of the time series
        prediction_length
            Length of the prediction horizon
        num_bins, optional
            Number ofum bins used for quantization of the time series,
            by default 1024
        num_residual_channels, optional
            Number of residual channels in wavenet architecture, by default 24
        num_skip_channels, optional
            Number of skip channels in wavenet architecture, by default 32
        dilation_depth, optional
            Number of dilation layers in wavenet architecture. If set to None
            (default), dialation_depth is set such that the receptive length
            is at least as long as typical seasonality for the frequency and
            at least 2 * prediction_length, by default None
        num_stacks, optional
            Number of dilation stacks in wavenet architecture, by default 1
        temperature, optional
            Temparature used for sampling from softmax distribution,
            by default 1.0
        num_feat_dynamic_real, optional
            The number of dynamic real features, by default 0
        num_feat_static_cat, optional
            The number of static categorical features, by default 0
        num_feat_static_real, optional
            The number of static real features, by default 0
        cardinality, optional
            The cardinalities of static categorical features, by default [1]
        seasonality, optional
            The seasonality of the time series. If set to None, seasonality
            is set to default value based on the `freq`, by default None
        embedding_dimension, optional
            The dimension of the embeddings for categorical features,
            by default 5
        use_log_scale_feature, optional
            If True, logarithm of the scale of the past data will be used as an
            additional static feature,
            by default True
        time_features, optional
            List of time features, from :py:mod:`gluonts.time_feature`,
            by default None
        lr, optional
            Learning rate, by default 1e-3
        weight_decay, optional
            Weight decay, by default 1e-8
        train_sampler, optional
            Controls the sampling of windows during training,
            by default None
        validation_sampler, optional
            Controls the sampling of windows during validation,
            by default None
        batch_size, optional
            The size of the batches to be used for training, by default 32
        num_batches_per_epoch, optional
            Number of batches to be processed in each training epoch,
            by default 50
        num_parallel_samples, optional
            The number of parallel samples to generate during inference.
            This parameter is only used in inference mode, by default 100
        negative_data, optional
            Flag indicating whether the time series take negative values,
            by default False
        trainer_kwargs, optional
            Additional arguments to provide to ``pl.Trainer`` for construction,
            by default None
        """
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.freq = freq
        self.prediction_length = prediction_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_bins = num_bins
        self.num_residual_channels = num_residual_channels
        self.num_skip_channels = num_skip_channels
        self.num_stacks = num_stacks
        self.use_log_scale_feature = use_log_scale_feature
        self.time_features = unwrap_or(
            time_features, time_features_from_frequency_str(freq)
        )
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_sampler = unwrap_or(
            train_sampler,
            ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.prediction_length
            ),
        )
        self.validation_sampler = unwrap_or(
            validation_sampler,
            ValidationSplitSampler(min_future=self.prediction_length),
        )
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_parallel_samples = num_parallel_samples
        self.negative_data = negative_data
        low = -10.0 if self.negative_data else 0
        high = 10.0
        bin_centers = np.linspace(low, high, self.num_bins)
        bin_edges = np.concatenate(
            [[-1e20], (bin_centers[1:] + bin_centers[:-1]) / 2.0, [1e20]]
        )
        self.bin_centers = bin_centers.tolist()
        self.bin_edges = bin_edges.tolist()

        self.seasonality = unwrap_or(
            seasonality,
            get_seasonality(
                freq,
                {
                    "H": 7 * 24,
                    "D": 7,
                    "W": 52,
                    "M": 12,
                    "B": 7 * 5,
                    "T": 24 * 60,
                    "S": 60 * 60,
                },
            ),
        )

        goal_receptive_length = max(
            2 * self.seasonality, 2 * self.prediction_length
        )
        if dilation_depth is None:
            d = 1
            while (
                WaveNet.get_receptive_field(
                    dilation_depth=d, num_stacks=num_stacks
                )
                < goal_receptive_length
            ):
                d += 1
            self.dilation_depth = d
        else:
            self.dilation_depth = dilation_depth
        self.context_length = WaveNet.get_receptive_field(
            dilation_depth=self.dilation_depth, num_stacks=num_stacks
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        return Chain(
            [
                RemoveFields(field_names=remove_field_names),
                (
                    SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])
                    if self.num_feat_static_cat == 0
                    else Identity()
                ),
                (
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                    if self.num_feat_static_real == 0
                    else Identity()
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=int
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=np.float32,
                ),
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
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
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                ),
                AsNumpyArray(
                    FieldName.FEAT_TIME, expected_ndim=2, dtype=np.float32
                ),
            ]
        )

    def _create_instance_splitter(self, mode: str):
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
            output_NTC=False,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + QuantizeMeanScaled(bin_edges=self.bin_edges)

    def create_training_data_loader(
        self,
        data: Dataset,
        module: WaveNetLightningModule,
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
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self, data: Dataset, module: WaveNetLightningModule, **kwargs
    ) -> Iterable:
        instances = self._create_instance_splitter("validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_lightning_module(self) -> pl.LightningModule:
        return WaveNetLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            model_kwargs=dict(
                bin_values=self.bin_centers,
                num_residual_channels=self.num_residual_channels,
                num_skip_channels=self.num_skip_channels,
                dilation_depth=self.dilation_depth,
                num_stacks=self.num_stacks,
                num_feat_dynamic_real=1
                + self.num_feat_dynamic_real
                + len(self.time_features),
                num_feat_static_real=max(1, self.num_feat_static_real),
                cardinality=self.cardinality,
                embedding_dimension=self.embedding_dimension,
                pred_length=self.prediction_length,
                num_parallel_samples=self.num_parallel_samples,
                temperature=self.temperature,
                use_log_scale_feature=self.use_log_scale_feature,
            ),
        )

    def create_predictor(
        self, transformation: Transformation, module: WaveNetLightningModule
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )
