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

import logging
import re
from functools import partial
from typing import Dict, List, Optional

import mxnet as mx
import numpy as np

from gluonts import transform
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor
from gluonts.model.wavenet._network import (
    WaveNet,
    WaveNetTraining,
    WaveNetSampler,
)
from gluonts.mx.batchify import batchify, as_in_context
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.time_feature import (
    get_seasonality,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SelectFields,
    SetFieldIfNotPresent,
    SimpleTransformation,
    VstackFeatures,
    InstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
)


class QuantizeScaled(SimpleTransformation):
    """
    Rescale and quantize the target variable.

    Requires
      past_target and future_target fields.

    The mean absolute value of the past_target is used to rescale past_target and future_target.
    Then the bin_edges are used to quantize the rescaled target.

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
        data[self.scale] = np.array([scale])
        return data


class WaveNetEstimator(GluonEstimator):
    """
    Model with Wavenet architecture and quantized target.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    trainer
        Trainer object to be used (default: Trainer())
    cardinality
        Number of values of the each categorical feature (default: [1])
    embedding_dimension
        Dimension of the embeddings for categorical features (the same
        dimension is used for all embeddings, default: 5)
    num_bins
        Number of bins used for quantization of signal (default: 1024)
    hybridize_prediction_net
        Boolean (default: False)
    n_residue
        Number of residual channels in wavenet architecture (default: 24)
    n_skip
        Number of skip channels in wavenet architecture (default: 32)
    dilation_depth
        Number of dilation layers in wavenet architecture.
        If set to None (default), dialation_depth is set such that the receptive length is at least
        as long as typical seasonality for the frequency and at least 2 * prediction_length.
    n_stacks
        Number of dilation stacks in wavenet architecture (default: 1)
    temperature
        Temparature used for sampling from softmax distribution.
        For temperature = 1.0 (default) sampling is according to estimated probability.
    act_type
        Activation type used after before output layer (default: "elu").
        Can be any of 'elu', 'relu', 'sigmoid', 'tanh', 'softrelu', 'softsign'.
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 200)
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    batch_size
        The size of the batches to be used training and prediction.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        trainer: Trainer = Trainer(
            learning_rate=0.01,
            epochs=200,
            num_batches_per_epoch=50,
            hybridize=False,
        ),
        cardinality: List[int] = [1],
        seasonality: Optional[int] = None,
        embedding_dimension: int = 5,
        num_bins: int = 1024,
        hybridize_prediction_net: bool = False,
        n_residue=24,
        n_skip=32,
        dilation_depth: Optional[int] = None,
        n_stacks: int = 1,
        train_window_length: Optional[int] = None,
        temperature: float = 1.0,
        act_type: str = "elu",
        num_parallel_samples: int = 200,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        batch_size: int = 32,
        negative_data: bool = False,
    ) -> None:
        super().__init__(trainer=trainer, batch_size=batch_size)

        self.freq = freq
        self.prediction_length = prediction_length
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_bins = num_bins
        self.hybridize_prediction_net = hybridize_prediction_net

        self.n_residue = n_residue
        self.n_skip = n_skip
        self.n_stacks = n_stacks
        self.train_window_length = (
            train_window_length
            if train_window_length is not None
            else prediction_length
        )
        self.temperature = temperature
        self.act_type = act_type
        self.num_parallel_samples = num_parallel_samples
        self.train_sampler = (
            train_sampler
            if train_sampler is not None
            else ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.train_window_length
            )
        )
        self.validation_sampler = (
            validation_sampler
            if validation_sampler is not None
            else ValidationSplitSampler(min_future=self.train_window_length)
        )
        self.negative_data = negative_data

        low = -10.0 if self.negative_data else 0
        high = 10.0
        bin_centers = np.linspace(low, high, self.num_bins)
        bin_edges = np.concatenate(
            [[-1e20], (bin_centers[1:] + bin_centers[:-1]) / 2.0, [1e20]]
        )
        self.bin_centers = bin_centers.tolist()
        self.bin_edges = bin_edges.tolist()

        seasonality = (
            get_seasonality(
                self.freq,
                {
                    "H": 7 * 24,
                    "D": 7,
                    "W": 52,
                    "M": 12,
                    "B": 7 * 5,
                    "min": 24 * 60,
                },
            )
            if seasonality is None
            else seasonality
        )

        goal_receptive_length = max(
            2 * seasonality, 2 * self.prediction_length
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
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Using dilation depth {self.dilation_depth} and receptive field length {self.context_length}"
        )

    def create_transformation(self) -> transform.Transformation:
        return Chain(
            [
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
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
            future_length=self.prediction_length
            if mode is "test"
            else self.train_window_length,
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
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(WaveNetTraining)
        instance_splitter = self._create_instance_splitter("training")
        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            decode_fn=partial(as_in_context, ctx=self.trainer.ctx),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(WaveNetTraining)
        instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def _get_wavenet_args(self):
        return dict(
            n_residue=self.n_residue,
            n_skip=self.n_skip,
            dilation_depth=self.dilation_depth,
            n_stacks=self.n_stacks,
            act_type=self.act_type,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            bin_values=self.bin_centers,
            pred_length=self.prediction_length,
        )

    def create_training_network(self) -> WaveNetTraining:
        params = self._get_wavenet_args()
        params.update(pred_length=self.train_window_length)
        return WaveNetTraining(**params)

    def create_predictor(
        self,
        transformation: transform.Transformation,
        trained_network: mx.gluon.HybridBlock,
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = WaveNetSampler(
            num_samples=self.num_parallel_samples,
            temperature=self.temperature,
            **self._get_wavenet_args(),
        )

        # The lookup layer is specific to the sampling network here
        # we make sure it is initialized.
        prediction_network.initialize()

        copy_parameters(
            net_source=trained_network,
            net_dest=prediction_network,
            allow_missing=True,
        )

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
