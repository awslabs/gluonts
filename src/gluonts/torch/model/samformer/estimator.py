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

from typing import Any, Dict, Iterable, Optional

import lightning.pytorch as pl
import torch

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.torch.distributions import Output, StudentTOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)

from .lightning_module import SamFormerLightningModule

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class SamFormerEstimator(PyTorchLightningEstimator):
    """
    An estimator training the SamFormer model for multivariate forecasting
    as described in TODO extended to be
    probabilistic.

    This class uses the model defined in ``SamFormerModel``,
    and wraps it into a ``SamFormerLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    hidden_dim
        Size of query and key projection (default: ``32``).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    scaling
        Scaling parameter can be "mean", "std" or None.
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    num_parallel_samples
        Number of samples per time series to that the resulting predictor
        should produce (default: 100).
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
    nonnegative_pred_samples
        Should final prediction samples be non-negative? If yes, an activation
        function is applied to ensure non-negative. Observe that this is applied
        only to the final samples and this is not applied during training.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: Optional[int] = None,
        hidden_dim: int = 32,
        projection_dim: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        rho: float = 0.5,
        sam: bool = True,
        scaling: Optional[str] = "mean",
        distr_output: Output = StudentTOutput(),
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        nonnegative_pred_samples: bool = False,
    ) -> None:
        default_trainer_kwargs = {"max_epochs": 100}

        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.prediction_length = prediction_length
        self.context_length = context_length or 10 * prediction_length
        # TODO find way to enforce same defaults to network and estimator
        # somehow
        self.lr = lr
        self.weight_decay = weight_decay
        self.rho = rho
        self.distr_output = distr_output
        self.num_parallel_samples = num_parallel_samples
        self.scaling = scaling
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.sam = sam
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.nonnegative_pred_samples = nonnegative_pred_samples

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        return (
            SelectFields(
                [
                    FieldName.ITEM_ID,
                    FieldName.INFO,
                    FieldName.START,
                    FieldName.TARGET,
                ],
                allow_missing=True,
            )
            + AsNumpyArray(field=FieldName.TARGET, expected_ndim=2)
            + AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            )
        )

    def create_lightning_module(self) -> pl.LightningModule:
        return SamFormerLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            rho=self.rho,
            num_parallel_samples=self.num_parallel_samples,
            sam=self.sam,
            model_kwargs={
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "hidden_dim": self.hidden_dim,
                "projection_dim": self.projection_dim,
                "distr_output": self.distr_output,
                "scaling": self.scaling,
                "nonnegative_pred_samples": self.nonnegative_pred_samples,
            },
        )

    def _create_instance_splitter(
        self, module: SamFormerLightningModule, mode: str
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
            time_series_fields=[FieldName.OBSERVED_VALUES],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: SamFormerLightningModule,
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
        self, data: Dataset, module: SamFormerLightningModule, **kwargs
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
        self, transformation: Transformation, module
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )
