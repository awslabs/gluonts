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

from typing import Optional, Iterable, Dict, Any

import torch
import lightning.pytorch as pl

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.transform import (
    AsNumpyArray,
    Transformation,
    AddObservedValuesIndicator,
    InstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import Output, StudentTOutput

from .lightning_module import ITransformerLightningModule

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class ITransformerEstimator(PyTorchLightningEstimator):
    """
    An estimator training the iTransformer model for multivariate forecasting
    as described in https://arxiv.org/abs/2310.06625 extended to be
    probabilistic.

    This class uses the model defined in ``ITransformerModel``,
    and wraps it into a ``ITransformerLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    d_model
        Size of latent in the Transformer encoder.
    nhead
        Number of attention heads in the Transformer encoder which must divide d_model.
    dim_feedforward
        Size of hidden layers in the Transformer encoder.
    dropout
        Dropout probability in the Transformer encoder.
    activation
        Activation function in the Transformer encoder.
    norm_first
        Whether to apply normalization before or after the attention.
    num_encoder_layers
        Number of layers in the Transformer encoder.
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
        d_model: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
        num_encoder_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
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
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.prediction_length = prediction_length
        self.context_length = context_length or 10 * prediction_length
        # TODO find way to enforce same defaults to network and estimator
        # somehow
        self.lr = lr
        self.weight_decay = weight_decay
        self.distr_output = distr_output
        self.num_parallel_samples = num_parallel_samples
        self.scaling = scaling
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.num_encoder_layers = num_encoder_layers
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
        return ITransformerLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            num_parallel_samples=self.num_parallel_samples,
            model_kwargs={
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "activation": self.activation,
                "norm_first": self.norm_first,
                "num_encoder_layers": self.num_encoder_layers,
                "distr_output": self.distr_output,
                "scaling": self.scaling,
                "nonnegative_pred_samples": self.nonnegative_pred_samples,
            },
        )

    def _create_instance_splitter(
        self, module: ITransformerLightningModule, mode: str
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
        module: ITransformerLightningModule,
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
        self, data: Dataset, module: ITransformerLightningModule, **kwargs
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
