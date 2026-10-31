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

from typing import Any, Dict, Iterable, List, Optional

import lightning.pytorch as pl
import torch

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.distributions import Output, StudentTOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)

from .lightning_module import CrossformerLightningModule

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class CrossformerEstimator(PyTorchLightningEstimator):
    """
    An estimator training the Crossformer model for forecasting as described in
    https://openreview.net/forum?id=vSVLM2j9eie, extended to be probabilistic.

    This class uses the model defined in ``CrossformerModel`` and wraps it in a
    ``CrossformerLightningModule`` for training with PyTorch Lightning's
    ``pl.Trainer``.

    Parameters
    ----------
    freq
        Frequency string of the data (used to build calendar time features).
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time used as input
        (default: ``10 * prediction_length``).
    seg_len
        Segment length for the dimension-segment-wise (DSW) embedding and
        multi-scale encoder.
    win_size
        Number of adjacent segments merged per scale in the encoder after the
        first scale (default: 2, as in the paper).
    factor
        Number of learnable router vectors in each two-stage attention layer.
    d_model
        Hidden size of the transformer blocks.
    d_ff
        Hidden size of the feed-forward sublayers.
    n_heads
        Number of attention heads (must be compatible with ``d_model``).
    num_encoder_layers
        Number of encoder scales (depth of the cross-scale hierarchy).
    num_feat_dynamic_real
        Number of dynamic real covariates in the data, stacked with calendar
        features (default: 0).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization (default: ``1e-8``).
    scaling
        Input scaling: ``"mean"``, ``"std"``, or ``None`` for no scaling.
    distr_output
        Distribution head for likelihood and sampling (default:
        ``StudentTOutput()``).
    num_parallel_samples
        Number of sample paths per series when the output defines a
        distribution (default: 100).
    batch_size
        Training and inference batch size (default: 32).
    num_batches_per_epoch
        Number of training batches per epoch (default: 50).
    trainer_kwargs
        Additional keyword arguments passed to ``pl.Trainer``.
    train_sampler
        Sampler for training windows (default: ``ExpectedNumInstanceSampler``).
    validation_sampler
        Sampler for validation windows (default: ``ValidationSplitSampler``).
    dropout
        Dropout probability in attention and feed-forward blocks.
    time_features
        Optional list of calendar features; if ``None``, derived from ``freq``.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        seg_len: int = 6,
        win_size: int = 2,
        factor: int = 10,
        d_model: int = 64,
        d_ff: int = 128,
        n_heads: int = 4,
        num_encoder_layers: int = 3,
        num_feat_dynamic_real: int = 0,
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
        dropout: float = 0.1,
        time_features: Optional[List[TimeFeature]] = None,
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
        self.context_length = context_length or 10 * prediction_length
        self.seg_len = seg_len
        self.win_size = win_size
        self.factor = factor
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.lr = lr
        self.weight_decay = weight_decay
        self.scaling = scaling
        self.distr_output = distr_output
        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.dropout = dropout
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )
        self.num_time_features = len(self.time_features) + self.num_feat_dynamic_real

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + [
                SelectFields(
                    [
                        FieldName.ITEM_ID,
                        FieldName.INFO,
                        FieldName.START,
                        FieldName.TARGET,
                    ]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                    allow_missing=True,
                ),
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=2),
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
            + (
                [
                    VstackFeatures(
                        output_field=FieldName.FEAT_TIME,
                        input_fields=[
                            FieldName.FEAT_TIME,
                            FieldName.FEAT_DYNAMIC_REAL,
                        ],
                    )
                ]
                if self.num_feat_dynamic_real > 0
                else []
            )
            + [
                AsNumpyArray(field=FieldName.FEAT_TIME, expected_ndim=2),
            ]
        )

    def create_lightning_module(self) -> pl.LightningModule:
        return CrossformerLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            num_parallel_samples=self.num_parallel_samples,
            model_kwargs={
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "seg_len": self.seg_len,
                "win_size": self.win_size,
                "factor": self.factor,
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "n_heads": self.n_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "num_feat_dynamic_real": self.num_time_features,
                "scaling": self.scaling,
                "distr_output": self.distr_output,
                "dropout": self.dropout,
            },
        )

    def _create_instance_splitter(
        self, module: CrossformerLightningModule, mode: str
    ) -> InstanceSplitter:
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
            time_series_fields=[FieldName.OBSERVED_VALUES]
            + ([FieldName.FEAT_TIME] if self.num_time_features > 0 else []),
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: CrossformerLightningModule,
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
            field_names=TRAINING_INPUT_NAMES
            + (
                [f"past_{FieldName.FEAT_TIME}", f"future_{FieldName.FEAT_TIME}"]
                if self.num_time_features > 0
                else []
            ),
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: CrossformerLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES
            + (
                [f"past_{FieldName.FEAT_TIME}", f"future_{FieldName.FEAT_TIME}"]
                if self.num_time_features > 0
                else []
            ),
            output_type=torch.tensor,
        )

    def create_predictor(
        self, transformation: Transformation, module
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")
        predictor_kwargs = dict(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES
            + (
                [f"past_{FieldName.FEAT_TIME}", f"future_{FieldName.FEAT_TIME}"]
                if self.num_time_features > 0
                else []
            ),
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )
        if not hasattr(self.distr_output, "distribution"):
            predictor_kwargs["forecast_generator"] = (
                self.distr_output.forecast_generator
            )
        return PyTorchPredictor(**predictor_kwargs)
