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

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.torch.util import copy_parameters
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    Transformation,
    Chain,
    ExpectedNumInstanceSampler,
    AddObservedValuesIndicator,
    AsNumpyArray,
)

from pts.model import PyTorchEstimator
from pts import Trainer
from pts.model.utils import get_module_forward_input_names

from .lstnet_network import LSTNetTrain, LSTNetPredict


class LSTNetEstimator(PyTorchEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: Optional[int],
        context_length: int,
        num_series: int,
        ar_window: int = 24,
        skip_size: int = 24,
        channels: int = 100,
        kernel_size: int = 6,
        horizon: Optional[int] = None,
        trainer: Trainer = Trainer(),
        dropout_rate: Optional[float] = 0.2,
        output_activation: Optional[str] = None,
        rnn_cell_type: str = "GRU",
        rnn_num_cells: int = 100,
        skip_rnn_cell_type: str = "GRU",
        skip_rnn_num_cells: int = 5,
        scaling: bool = True,
        dtype: np.dtype = np.float32,
    ):
        super().__init__(trainer, dtype=dtype)

        self.freq = freq
        self.num_series = num_series
        self.skip_size = skip_size
        self.ar_window = ar_window
        self.horizon = horizon
        self.prediction_length = prediction_length

        self.future_length = (
            horizon if horizon is not None else prediction_length
        )
        self.context_length = context_length
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.rnn_cell_type = rnn_cell_type
        self.rnn_num_cells = rnn_num_cells
        self.skip_rnn_cell_type = skip_rnn_cell_type
        self.skip_rnn_num_cells = skip_rnn_num_cells
        self.scaling = scaling

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=self.future_length
        )
        self.validation_sampler = ValidationSplitSampler(
            min_future=self.future_length
        )

        self.dtype = dtype

    def create_transformation(self) -> Transformation:
        return Chain(
            trans=[
                AsNumpyArray(
                    field=FieldName.TARGET, expected_ndim=2, dtype=self.dtype
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
            ]
        )

    def create_instance_splitter(self, mode: str):
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
            time_series_fields=[FieldName.OBSERVED_VALUES],
            past_length=self.context_length,
            future_length=self.future_length,
            output_NTC=False,
        )

    def create_training_network(self, device: torch.device) -> LSTNetTrain:
        return LSTNetTrain(
            num_series=self.num_series,
            channels=self.channels,
            kernel_size=self.kernel_size,
            rnn_cell_type=self.rnn_cell_type,
            rnn_num_cells=self.rnn_num_cells,
            skip_rnn_cell_type=self.skip_rnn_cell_type,
            skip_rnn_num_cells=self.skip_rnn_num_cells,
            skip_size=self.skip_size,
            ar_window=self.ar_window,
            context_length=self.context_length,
            horizon=self.horizon,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            output_activation=self.output_activation,
            scaling=self.scaling,
        ).to(device)

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: LSTNetTrain,
        device: torch.device,
    ) -> PyTorchPredictor:
        prediction_network = LSTNetPredict(
            num_series=self.num_series,
            channels=self.channels,
            kernel_size=self.kernel_size,
            rnn_cell_type=self.rnn_cell_type,
            rnn_num_cells=self.rnn_num_cells,
            skip_rnn_cell_type=self.skip_rnn_cell_type,
            skip_rnn_num_cells=self.skip_rnn_num_cells,
            skip_size=self.skip_size,
            ar_window=self.ar_window,
            context_length=self.context_length,
            horizon=self.horizon,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            output_activation=self.output_activation,
            scaling=self.scaling,
        ).to(device)

        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.horizon or self.prediction_length,
            device=device,
        )
