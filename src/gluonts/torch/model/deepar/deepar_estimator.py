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

from typing import List, Optional, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
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
    get_forward_input_names,
    IterableDataset,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    StudentTOutput,
)

from .deepar_network import (
    DeepARLightningNetwork,
    DeepARNetwork,
)


class DeepAREstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        trainer: pl.Trainer = pl.Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "LSTM",
        dropout_rate: float = 0.1,
        num_feat_dynamic_real: int = 0,
        num_feat_dynamic_cat: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        dtype: np.dtype = np.float32,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
    ) -> None:
        super().__init__(trainer=trainer)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.distr_output.dtype = dtype
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_dynamic_cat = num_feat_dynamic_cat
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )
        self.scaling = scaling
        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_lags_for_frequency(
                freq_str=freq, lag_ub=self.context_length
            )
        )
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)

        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if self.num_feat_dynamic_cat == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_CAT)

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
                    dtype=np.long,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
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
                        if self.num_feat_dynamic_real > 0
                        else []
                    )
                    + (
                        [FieldName.FEAT_DYNAMIC_CAT]
                        if self.num_feat_dynamic_cat > 0
                        else []
                    ),
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
            past_length=self.history_length,
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
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        input_names = get_forward_input_names(DeepARNetwork)
        transformation = self._create_instance_splitter(
            "training"
        ) + SelectFields(input_names)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            DataLoader(
                IterableDataset(training_instances),
                batch_size=self.batch_size,
                **kwargs,
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> Iterable:
        input_names = get_forward_input_names(DeepARNetwork)
        transformation = self._create_instance_splitter(
            "validation"
        ) + SelectFields(input_names)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_network(self) -> DeepARLightningNetwork:
        return DeepARLightningNetwork(
            freq=self.freq,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=1
            + self.num_feat_dynamic_real
            + len(self.time_features),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            context_length=self.context_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        network: DeepARNetwork,
        device: torch.device,
    ) -> PyTorchPredictor:
        input_names = get_forward_input_names(DeepARNetwork)
        prediction_splitter = self._create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=network,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
