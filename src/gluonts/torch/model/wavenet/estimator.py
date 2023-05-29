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

import pytorch_lightning as pl
import torch
import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset
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
    SimpleTransformation,
    VstackFeatures,
    Identity,
    TestSplitSampler,
    ValidationSplitSampler,
)

from .module import WaveNet
from .lightning_module import WaveNetLightningModule

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
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


class QuantizeScaled(SimpleTransformation):
    """
    Rescale and quantize the target variable.

    Requires
      past_target and future_target fields.

    The mean absolute value of the past_target is used to rescale past_target
    and future_target. Then the bin_edges are used to quantize the rescaled
    target.

    The calculated scale is included as a new field "scale"
    """

    @validated()
    def __init__(
        self,
        bin_edges: List[float],
        past_target: str,
        future_target: str,
        scale: str = "scale",
    ):
        self.bin_edges = np.array(bin_edges)
        self.future_target = future_target
        self.past_target = past_target
        self.scale = scale

    def transform(self, data: DataEntry) -> DataEntry:
        p = data[self.past_target]
        m = np.mean(np.abs(p))
        scale = m if m > 0 else 1.0
        data[self.future_target] = np.digitize(
            data[self.future_target] / scale, bins=self.bin_edges, right=False
        )
        data[self.past_target] = np.digitize(
            data[self.past_target] / scale, bins=self.bin_edges, right=False
        )
        data[self.scale] = np.array([scale], dtype=np.float32)
        return data


class WaveNetEstimator(PyTorchLightningEstimator):
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        n_bins: int = 1024,
        n_residual_channels: int = 24,
        n_skip_channels: int = 32,
        dilation_depth: Optional[int] = None,
        n_stacks: int = 1,
        temperature: float = 1.0,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: List[int] = [1],
        seasonality: Optional[int] = None,
        embedding_dimension: int = 5,
        time_features: Optional[List[TimeFeature]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        num_parallel_samples: int = 100,
        negative_data: bool = False,
        trainer_kwargs: Dict[str, Any] = None,
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
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.n_bins = n_bins
        self.n_residual_channels = n_residual_channels
        self.n_skip_channels = n_skip_channels
        self.n_stacks = n_stacks
        self.time_features = time_features or time_features_from_frequency_str(
            freq
        )
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=self.prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=self.prediction_length
        )
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_parallel_samples = num_parallel_samples
        self.negative_data = negative_data
        low = -10.0 if self.negative_data else 0
        high = 10.0
        bin_centers = np.linspace(low, high, self.n_bins)
        bin_edges = np.concatenate(
            [[-1e20], (bin_centers[1:] + bin_centers[:-1]) / 2.0, [1e20]]
        )
        self.bin_centers = bin_centers.tolist()
        self.bin_edges = bin_edges.tolist()

        self.seasonality = seasonality or get_seasonality(
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
        )

        goal_receptive_length = max(
            2 * self.seasonality, 2 * self.prediction_length
        )
        if dilation_depth is None:
            d = 1
            while (
                WaveNet.get_receptive_field(
                    dilation_depth=d, n_stacks=n_stacks
                )
                < goal_receptive_length
            ):
                d += 1
            self.dilation_depth = d
        else:
            self.dilation_depth = dilation_depth
        self.context_length = WaveNet.get_receptive_field(
            dilation_depth=self.dilation_depth, n_stacks=n_stacks
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
                SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])
                if not self.num_feat_static_cat > 0
                else Identity(),
                SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])
                if not self.num_feat_static_real > 0
                else Identity(),
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
        ) + QuantizeScaled(
            bin_edges=self.bin_edges,
            future_target="future_target",
            past_target="past_target",
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: WaveNetLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs
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
                n_residual_channels=self.n_residual_channels,
                n_skip_channels=self.n_skip_channels,
                dilation_depth=self.dilation_depth,
                n_stacks=self.n_stacks,
                num_feat_dynamic_real=1
                + self.num_feat_dynamic_real
                + len(self.time_features),
                num_feat_static_real=max(1, self.num_feat_static_real),
                cardinality=self.cardinality,
                embedding_dimension=self.embedding_dimension,
                pred_length=self.prediction_length,
                n_parallel_samples=self.num_parallel_samples,
                temperature=self.temperature,
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
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
