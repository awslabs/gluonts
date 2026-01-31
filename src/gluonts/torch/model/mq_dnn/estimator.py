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

import numpy as np
import torch

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.torch.distributions import QuantileOutput
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    AddConstFeature,
    VstackFeatures,
    TestSplitSampler,
    ValidationSplitSampler,
    ExpectedNumInstanceSampler,
    DummyValueImputation,
    AddSeriesScale,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform.sampler import InstanceSampler

# Import the framework-agnostic forking sequence splitter from MXNet
from gluonts.mx.model.seq2seq._transform import ForkingSequenceSplitter

from .lightning_module import MQDNNLightningModule
from .module import (
    HierarchicalCausalConv1DEncoder,
    RNNEncoder,
)


PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_feat_dynamic",  # Changed from past_time_feat to match FEAT_DYNAMIC
    "past_target",
    "past_observed_values",
    "future_feat_dynamic",  # Changed from future_time_feat to match FEAT_DYNAMIC
    "series_scale",  # Pre-computed series-level scale (before forking)
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class MQDNNEstimator(PyTorchLightningEstimator):
    """
    Base estimator class for MQ-DNN models (Multi-Quantile Deep Neural Network).

    This class provides common functionality for both MQ-CNN and MQ-RNN variants.
    Do not instantiate this class directly; use MQCNNEstimator or MQRNNEstimator.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of steps for the encoder (default: 4 * prediction_length).
    num_feat_dynamic_real
        Number of dynamic real features in the data (default: 0).
    num_feat_static_cat
        Number of static categorical features in the data (default: 0).
    num_feat_static_real
        Number of static real features in the data (default: 0).
    cardinality
        Number of values of each categorical feature.
    embedding_dimension
        Dimension of the embeddings for categorical features.
    add_time_feature
        Whether to add time features (default: True).
    add_age_feature
        Whether to add age feature (default: False).
    encoder
        Encoder module (CNN or RNN).
    decoder_mlp_dim_seq
        Sequence of MLP dimensions for the decoder (default: [30]).
    quantiles
        List of quantiles to predict.
    scaling
        Whether to automatically scale the target values (default: True).
    num_forking
        Number of forking positions (default: context_length).
    lr
        Learning rate (default: 1e-3).
    weight_decay
        Weight decay regularization parameter (default: 1e-8).
    patience
        Patience parameter for learning rate scheduler (default: 10).
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch (default: 50).
    trainer_kwargs
        Additional arguments to provide to pl.Trainer for construction.
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
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        add_time_feature: bool = True,
        add_age_feature: bool = False,
        encoder=None,
        decoder_mlp_dim_seq: Optional[List[int]] = None,
        quantiles: Optional[List[float]] = None,
        scaling: Optional[bool] = None,  # Default: False for quantile output (matches MXNet)
        num_forking: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
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
        self.context_length = (
            context_length
            if context_length is not None
            else 4 * prediction_length
        )
        self.prediction_length = prediction_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = embedding_dimension
        self.add_time_feature = add_time_feature
        self.add_age_feature = add_age_feature
        self.encoder = encoder
        self.decoder_mlp_dim_seq = decoder_mlp_dim_seq or [30]
        self.quantiles = quantiles or [
            0.025,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.975,
        ]
        # Match MXNet behavior: default to False for quantile output (NOPScaler)
        # MXNet: scaling = (scaling if scaling is not None else (quantile_output is None))
        # For quantile output (our case), this evaluates to False
        self.scaling = scaling if scaling is not None else False
        self.num_forking = (
            num_forking if num_forking is not None else self.context_length
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.time_features = (
            time_features_from_frequency_str(self.freq)
            if add_time_feature
            else []
        )

        self.train_sampler = train_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "num_feat_dynamic_real": stats.num_feat_dynamic_real,
            "num_feat_static_cat": len(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def create_transformation(self) -> Transformation:
        """
        Create the transformation pipeline for preprocessing data.
        """
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]

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
                    expected_ndim=1,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    imputation_method=DummyValueImputation(0.0),
                ),
            ]
            + (
                [
                    AddSeriesScale(
                        target_field=FieldName.TARGET,
                        observed_field=FieldName.OBSERVED_VALUES,
                        scale_field="series_scale",
                        minimum_scale=1e-10,
                    ),
                ]
                if self.scaling
                else [
                    # When scaling=False (NOPScaler), set scale to 1.0 as float32
                    # to match the dtype used by AddSeriesScale
                    SetField(output_field="series_scale", value=np.float32(1.0)),
                ]
            )
            + [
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
            ]
            + (
                [
                    AddAgeFeature(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_AGE,
                        pred_length=self.prediction_length,
                        log_scale=True,
                    )
                ]
                if self.add_age_feature
                else []
            )
            + (
                [
                    # Vstack into FEAT_DYNAMIC to match MXNet
                    VstackFeatures(
                        output_field=FieldName.FEAT_DYNAMIC,
                        input_fields=[FieldName.FEAT_TIME]
                        + ([FieldName.FEAT_AGE] if self.add_age_feature else [])
                        + (
                            [FieldName.FEAT_DYNAMIC_REAL]
                            if self.num_feat_dynamic_real > 0
                            else []
                        ),
                    ),
                    AsNumpyArray(FieldName.FEAT_DYNAMIC, expected_ndim=2),
                ]
                # Only add VstackFeatures if there are features to stack
                if len(self.time_features) > 0 or self.add_age_feature or self.num_feat_dynamic_real > 0
                else [
                    # When no features, create a dummy constant feature
                    AddConstFeature(
                        output_field=FieldName.FEAT_DYNAMIC,
                        target_field=FieldName.TARGET,
                        pred_length=self.prediction_length,
                        const=0.0,
                    ),
                    AsNumpyArray(FieldName.FEAT_DYNAMIC, expected_ndim=2),
                ]
            )
        )

    def _create_instance_splitter(
        self, module: MQDNNLightningModule, mode: str
    ):
        """
        Create the instance splitter with forking sequence support.
        """
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return ForkingSequenceSplitter(
            target_field=FieldName.TARGET,
            is_pad_out=FieldName.IS_PAD,
            start_input_field=FieldName.START,
            instance_sampler=instance_sampler,
            enc_len=self.context_length,
            dec_len=self.prediction_length,
            # Use FEAT_DYNAMIC like MXNet, not FEAT_TIME
            encoder_series_fields=[FieldName.OBSERVED_VALUES, FieldName.FEAT_DYNAMIC],
            decoder_series_fields=[FieldName.OBSERVED_VALUES, FieldName.FEAT_DYNAMIC],
            encoder_disabled_fields=[],
            decoder_disabled_fields=[],
            prediction_time_decoder_exclude=[FieldName.OBSERVED_VALUES],
            num_forking=self.num_forking,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: MQDNNLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        """
        Create training data loader with forking sequence support.
        """
        transformation = self._create_instance_splitter(module, "training")

        data = Cyclic(data).stream()
        instances = transformation.apply(data, is_train=True)

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
        module: MQDNNLightningModule,
        **kwargs,
    ) -> Iterable:
        """
        Create validation data loader with forking sequence support.
        """
        transformation = self._create_instance_splitter(module, "validation")

        instances = transformation.apply(data, is_train=True)

        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_lightning_module(self) -> MQDNNLightningModule:
        """
        Create the Lightning module for training.
        """
        # Count actual dynamic features created by transformation:
        # - time_features (based on frequency)
        # - age feature (only if add_age_feature=True)
        # - user-provided feat_dynamic_real (if any)
        # - dummy constant feature (if no other features exist)
        num_dynamic_features = (
            len(self.time_features)
            + (1 if self.add_age_feature else 0)  # age feature (conditional)
            + self.num_feat_dynamic_real  # user-provided dynamic features
        )

        # If no features at all, we add a dummy constant feature
        if num_dynamic_features == 0:
            num_dynamic_features = 1

        model_kwargs = {
            "freq": self.freq,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "num_feat_dynamic_real": num_dynamic_features,
            "num_feat_static_cat": max(self.num_feat_static_cat, 1),
            "num_feat_static_real": max(self.num_feat_static_real, 1),
            "cardinality": self.cardinality,
            "embedding_dimension": self.embedding_dimension,
            "encoder": self.encoder,
            "decoder_mlp_dim_seq": self.decoder_mlp_dim_seq,
            "quantiles": self.quantiles,
            "scaling": self.scaling,
            "num_forking": self.num_forking,
        }

        return MQDNNLightningModule(
            model_kwargs=model_kwargs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: MQDNNLightningModule,
    ) -> PyTorchPredictor:
        """
        Create a predictor from the trained module.
        """
        prediction_splitter = self._create_instance_splitter(module, "test")

        # Use QuantileOutput to generate QuantileForecast objects
        quantile_output = QuantileOutput(self.quantiles)

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            forecast_generator=quantile_output.forecast_generator,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )


class MQCNNEstimator(MQDNNEstimator):
    """
    Estimator for MQ-CNN (Multi-Quantile Convolutional Neural Network).

    Uses a hierarchical causal CNN as the encoder with dilated convolutions.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of steps for the encoder (default: 4 * prediction_length).
    channels_seq
        Number of channels for each convolutional layer (default: [30, 30, 30]).
    dilation_seq
        Dilation rates for each convolutional layer (default: [1, 3, 9]).
    kernel_size_seq
        Kernel sizes for each convolutional layer (default: [7, 3, 3]).
    use_residual
        Whether to use residual connections (default: True).
    decoder_mlp_dim_seq
        Sequence of MLP dimensions for the decoder (default: [30]).
    quantiles
        List of quantiles to predict.
    scaling
        Whether to automatically scale the target values (default: True).
    num_forking
        Number of forking positions (default: context_length).
    lr
        Learning rate (default: 1e-3).
    weight_decay
        Weight decay regularization parameter (default: 1e-8).
    patience
        Patience parameter for learning rate scheduler (default: 10).
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch (default: 50).
    trainer_kwargs
        Additional arguments to provide to pl.Trainer for construction.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        channels_seq: Optional[List[int]] = None,
        dilation_seq: Optional[List[int]] = None,
        kernel_size_seq: Optional[List[int]] = None,
        use_residual: bool = True,
        decoder_mlp_dim_seq: Optional[List[int]] = None,
        quantiles: Optional[List[float]] = None,
        scaling: Optional[bool] = None,  # Default: False for quantile output (matches MXNet)
        num_forking: Optional[int] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        add_time_feature: bool = True,
        add_age_feature: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        channels_seq = channels_seq or [30, 30, 30]
        dilation_seq = dilation_seq or [1, 3, 9]
        kernel_size_seq = kernel_size_seq or [7, 3, 3]

        assert (
            len(channels_seq) == len(dilation_seq) == len(kernel_size_seq)
        ), "channels_seq, dilation_seq, and kernel_size_seq must have the same length"

        # Use lazy initialization (input_channels=None) because transformations
        # add features to the data after estimator initialization
        encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=dilation_seq,
            kernel_size_seq=kernel_size_seq,
            channels_seq=channels_seq,
            use_residual=use_residual,
            input_channels=None,  # Lazy initialization with PyTorch lazy modules
        )

        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_cat=num_feat_static_cat,
            num_feat_static_real=num_feat_static_real,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            add_time_feature=add_time_feature,
            add_age_feature=add_age_feature,
            encoder=encoder,
            decoder_mlp_dim_seq=decoder_mlp_dim_seq,
            quantiles=quantiles,
            scaling=scaling,
            num_forking=num_forking,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            trainer_kwargs=trainer_kwargs,
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
        )


class MQRNNEstimator(MQDNNEstimator):
    """
    Estimator for MQ-RNN (Multi-Quantile Recurrent Neural Network).

    Uses a bidirectional RNN as the encoder.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of steps for the encoder (default: 4 * prediction_length).
    hidden_size
        Number of hidden units in the RNN (default: 50).
    num_layers
        Number of RNN layers (default: 1).
    bidirectional
        Whether to use bidirectional RNN (default: True).
    cell_type
        Type of RNN cell: 'lstm' or 'gru' (default: 'gru').
    decoder_mlp_dim_seq
        Sequence of MLP dimensions for the decoder (default: [30]).
    quantiles
        List of quantiles to predict.
    scaling
        Whether to automatically scale the target values (default: True).
    num_forking
        Number of forking positions (default: context_length).
    lr
        Learning rate (default: 1e-3).
    weight_decay
        Weight decay regularization parameter (default: 1e-8).
    patience
        Patience parameter for learning rate scheduler (default: 10).
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch (default: 50).
    trainer_kwargs
        Additional arguments to provide to pl.Trainer for construction.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        hidden_size: int = 50,
        num_layers: int = 1,
        bidirectional: bool = True,
        cell_type: str = "gru",
        decoder_mlp_dim_seq: Optional[List[int]] = None,
        quantiles: Optional[List[float]] = None,
        scaling: Optional[bool] = None,  # Default: False for quantile output (matches MXNet)
        num_forking: Optional[int] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        add_time_feature: bool = True,
        add_age_feature: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        # Use lazy initialization because transformations add features after estimator init
        encoder = RNNEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            cell_type=cell_type,
            input_size=None,  # Lazy initialization
        )

        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_cat=num_feat_static_cat,
            num_feat_static_real=num_feat_static_real,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            add_time_feature=add_time_feature,
            add_age_feature=add_age_feature,
            encoder=encoder,
            decoder_mlp_dim_seq=decoder_mlp_dim_seq,
            quantiles=quantiles,
            scaling=scaling,
            num_forking=num_forking,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            trainer_kwargs=trainer_kwargs,
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
        )
