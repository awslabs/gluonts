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

from functools import partial
from typing import List

import mxnet as mx
from pydantic import Field

from gluonts import transform
from gluonts.core import serde
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.env import env
from gluonts.model.forecast import Quantile
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.block.decoder import OneShotDecoder
from gluonts.mx.block.enc2dec import PassThroughEnc2Dec
from gluonts.mx.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    MLPEncoder,
    RNNEncoder,
    Seq2SeqEncoder,
)
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.block.scaler import NOPScaler, Scaler
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.itertools import maybe_len
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    InstanceSampler,
    SelectFields,
    TestSplitSampler,
    ValidationSplitSampler,
)

from ._seq2seq_network import Seq2SeqPredictionNetwork, Seq2SeqTrainingNetwork


@serde.dataclass
class Seq2SeqEstimator(GluonEstimator):
    """
    Quantile-Regression Sequence-to-Sequence Estimator.
    """

    freq: str
    prediction_length: int = Field(..., gt=0)
    cardinality: List[int] = Field(...)
    embedding_dimension: int = Field(...)
    encoder: Seq2SeqEncoder = Field(...)
    decoder_mlp_layer: List[int] = Field(...)
    decoder_mlp_static_dim: int = Field(...)
    scaler: Scaler = NOPScaler()
    context_length: int = Field(None, gt=0)
    quantiles: List[float] = Field(None, ge=0, le=1)
    trainer: Trainer = Trainer()
    train_sampler: InstanceSampler = Field(None)
    validation_sampler: InstanceSampler = Field(None)
    num_parallel_samples: int = 100
    batch_size: int = 32

    def __post_init_post_parse__(self):
        super().__init__(trainer=self.trainer, batch_size=self.batch_size)

        self.context_length = (
            self.context_length
            if self.context_length is not None
            else self.prediction_length
        )
        self.quantiles = (
            self.quantiles if self.quantiles is not None else [0.1, 0.5, 0.9]
        )
        self.embedder = FeatureEmbedder(
            cardinalities=self.cardinality,
            embedding_dims=[
                self.embedding_dimension for _ in self.cardinality
            ],
        )
        self.train_sampler = (
            self.train_sampler
            if self.train_sampler is not None
            else ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.prediction_length
            )
        )
        self.validation_sampler = (
            self.validation_sampler
            if self.validation_sampler is not None
            else ValidationSplitSampler(min_future=self.prediction_length)
        )

    def create_transformation(self) -> transform.Transformation:
        return transform.Chain(
            trans=[
                transform.AsNumpyArray(
                    field=FieldName.TARGET, expected_ndim=1
                ),
                transform.AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
                transform.VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                transform.SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                transform.AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT, expected_ndim=1
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

        return transform.InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.FEAT_DYNAMIC_REAL],
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(Seq2SeqTrainingNetwork)
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("training")
        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(Seq2SeqTrainingNetwork)
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> mx.gluon.HybridBlock:
        distribution = QuantileOutput(self.quantiles)

        enc2dec = PassThroughEnc2Dec()
        decoder = OneShotDecoder(
            decoder_length=self.prediction_length,
            layer_sizes=self.decoder_mlp_layer,
            static_outputs_per_time_step=self.decoder_mlp_static_dim,
        )

        training_network = Seq2SeqTrainingNetwork(
            embedder=self.embedder,
            scaler=self.scaler,
            encoder=self.encoder,
            enc2dec=enc2dec,
            decoder=decoder,
            quantile_output=distribution,
        )

        return training_network

    def create_predictor(
        self,
        transformation: transform.Transformation,
        trained_network: Seq2SeqTrainingNetwork,
    ) -> Predictor:
        # todo: this is specific to quantile output
        quantile_strs = [
            Quantile.from_float(quantile).name for quantile in self.quantiles
        ]

        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = Seq2SeqPredictionNetwork(
            embedder=trained_network.embedder,
            scaler=trained_network.scaler,
            encoder=trained_network.encoder,
            enc2dec=trained_network.enc2dec,
            decoder=trained_network.decoder,
            quantile_output=trained_network.quantile_output,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            forecast_generator=QuantileForecastGenerator(quantile_strs),
        )


# TODO: fix mutable arguments
@serde.dataclass
class MLP2QRForecaster(Seq2SeqEstimator):
    freq: str
    prediction_length: int
    cardinality: List[int]
    embedding_dimension: int
    encoder_mlp_layer: List[int] = Field(...)
    decoder_mlp_layer: List[int] = Field(...)
    decoder_mlp_static_dim: int = Field(...)
    scaler: Scaler = NOPScaler()
    context_length: int = Field(None)
    quantiles: List[float] = Field(None)
    trainer: Trainer = Trainer()
    num_parallel_samples: int = 100

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

        self.encoder = MLPEncoder(layer_sizes=self.encoder_mlp_layer)


@serde.dataclass
class RNN2QRForecaster(Seq2SeqEstimator):
    freq: str
    prediction_length: int
    cardinality: List[int]
    embedding_dimension: int
    encoder_rnn_layer: int = Field(...)
    encoder_rnn_num_hidden: int = Field(...)
    decoder_mlp_layer: List[int] = Field(...)
    decoder_mlp_static_dim: int = Field(...)
    encoder_rnn_model: str = "lstm"
    encoder_rnn_bidirectional: bool = True
    scaler: Scaler = NOPScaler()
    context_length: int = Field(None)
    quantiles: List[float] = Field(None)
    trainer: Trainer = Trainer()
    num_parallel_samples: int = 100

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

        self.encoder = RNNEncoder(
            mode=self.encoder_rnn_model,
            hidden_size=self.encoder_rnn_num_hidden,
            num_layers=self.encoder_rnn_layer,
            bidirectional=self.encoder_rnn_bidirectional,
            use_static_feat=True,
            use_dynamic_feat=True,
        )


@serde.dataclass
class CNN2QRForecaster(Seq2SeqEstimator):
    freq: str
    prediction_length: int
    cardinality: List[int]
    embedding_dimension: int
    decoder_mlp_layer: List[int] = Field(...)
    decoder_mlp_static_dim: int = Field(...)
    scaler: Scaler = NOPScaler()
    context_length: int = Field(None)
    quantiles: List[float] = Field(None)
    trainer: Trainer = Trainer()
    num_parallel_samples: int = 100

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

        self.encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=[1, 3, 9],
            kernel_size_seq=([3] * len([30, 30, 30])),
            channels_seq=[30, 30, 30],
            use_residual=True,
            use_dynamic_feat=True,
            use_static_feat=True,
        )
