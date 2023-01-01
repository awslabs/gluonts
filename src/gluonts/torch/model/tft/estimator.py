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

import torch
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
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
    IterableDataset,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform.sampler import InstanceSampler

from .module import TemporalFusionTransformerModel
from .lightning_module import TemporalFusionTransformerLightningModule
from .transformation import TFTInstanceSplitter


class TemporalFusionTransformerEstimator(PyTorchLightningEstimator):
    """
    Estimator class to train a Temporal Fusion Transformer model, as described in [SFG17]_.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length).
    quantiles
    num_heads
    hidden_size
    num_feat_static_real
    num_feat_dynamic_real
    num_past_feat_dynamic_real
    cardinalities_static
    embedding_dimension
    time_features
    lr
    dropout_rate
    patience
    scaling
    batch_size
    num_batches_per_epoch
    trainer_kwargs
    train_sampler
    validation_sampler
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        num_heads: int = 4,
        hidden_size: int = 40,
        num_feat_static_real: int = 0,
        num_feat_dynamic_real: int = 0,
        num_past_feat_dynamic_real: int = 0,
        cardinalities_static: Optional[List[int]] = None,
        cardinalities_dynamic: Optional[List[int]] = None,
        cardinalities_past_dynamic: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        lr: float = 1e-3,
        dropout_rate: float = 0.1,
        patience: int = 10,
        scaling: bool = True,
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
        self.quantiles = quantiles
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_real = num_feat_static_real
        self.num_past_feat_dynamic_real = num_past_feat_dynamic_real
        self.cardinalities_static = cardinalities_static
        self.cardinalities_dynamic = cardinalities_dynamic
        self.cardinalities_past_dynamic = cardinalities_past_dynamic
        self.embedding_dimension = embedding_dimension
        if time_features is None:
            time_features = time_features_from_frequency_str(self.freq)
        self.time_features = time_features

        # Training procedure
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.patience = patience
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

        transforms = Chain(
            [RemoveFields(field_names=remove_field_names)]
            + [
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
            ]
        )

        # TODO

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        ts_fields = [FieldName.FEAT_DYNAMIC_CAT, FieldName.FEAT_DYNAMIC_REAL]
        past_ts_fields = [
            FieldName.PAST_FEAT_DYNAMIC_CAT,
            FieldName.PAST_FEAT_DYNAMIC_REAL,
        ]

        return TFTInstanceSplitter(
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: TemporalFusionTransformerLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        # TODO
        pass

    def create_lightning_module(
        self,
    ) -> TemporalFusionTransformerLightningModule:
        model = TemporalFusionTransformerModel(
            freq=self.freq,
        )
        return TemporalFusionTransformerLightningModule(
            model=model,
            lr=self.lr,
            patience=self.patience,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: TemporalFusionTransformerLightningModule,
    ) -> PyTorchPredictor:
        # TODO
        prediction_splitter = self._create_instance_splitter(module, "test")

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
