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
from pydantic import Field

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gluonts.core import serde
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSampler,
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
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)

from .module import SimpleFeedForwardModel
from .lightning_module import SimpleFeedForwardLightningModule

PREDICTION_INPUT_NAMES = [
    "past_target",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


@serde.dataclass
class SimpleFeedForwardEstimator(PyTorchLightningEstimator):
    """
    An estimator training a feedforward model for forecasting.

    This class is uses the model defined in ``SimpleFeedForwardModel``,
    and wraps it into a ``SimpleFeedForwardLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    hidden_dimensions
        Size of hidden layers in the feedforward network
        (default: ``[20, 20]``).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    loss
        Loss to be optimized during training
        (default: ``NegativeLogLikelihood()``).
    batch_norm
        Whether to apply batch normalization.
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

    prediction_length: int = Field(...)
    context_length: int = Field(None)
    hidden_dimensions: List[int] = Field(None)
    distr_output: DistributionOutput = StudentTOutput()
    loss: DistributionLoss = NegativeLogLikelihood()
    batch_norm: bool = False
    batch_size: int = 32
    num_batches_per_epoch: int = 50
    trainer_kwargs: Dict[str, Any] = Field(None)
    train_sampler: InstanceSampler = Field(None)
    validation_sampler: InstanceSampler = Field(None)

    def __post_init_post_parse__(self):
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if self.trainer_kwargs is not None:
            default_trainer_kwargs.update(self.trainer_kwargs)
        self.trainer_kwargs = default_trainer_kwargs

        if self.context_length is None:
            self.context_length = 10 * self.prediction_length

        # TODO find way to enforce same defaults to network and estimator
        # somehow
        if self.hidden_dimensions is None:
            self.hidden_dimensions = [20, 20]

        if self.train_sampler is None:
            self.train_sampler = ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.prediction_length
            )

        if self.validation_sampler is None:
            self.validation_sampler = ValidationSplitSampler(
                min_future=self.prediction_length
            )

    def create_transformation(self) -> Transformation:
        return SelectFields(
            [
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
            ],
            allow_missing=True,
        ) + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    def create_lightning_module(self) -> pl.LightningModule:
        model = SimpleFeedForwardModel(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            hidden_dimensions=self.hidden_dimensions,
            distr_output=self.distr_output,
            batch_norm=self.batch_norm,
        )
        return SimpleFeedForwardLightningModule(model=model, loss=self.loss)

    def _create_instance_splitter(
        self, module: SimpleFeedForwardLightningModule, mode: str
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
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: SimpleFeedForwardLightningModule,
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
        module: SimpleFeedForwardLightningModule,
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
        module,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module.model,
            forecast_generator=DistributionForecastGenerator(
                self.distr_output
            ),
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
