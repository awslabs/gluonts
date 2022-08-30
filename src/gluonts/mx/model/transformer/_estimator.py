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

from mxnet.gluon import HybridBlock
from pydantic import Field

from gluonts.core import serde
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.env import env
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.itertools import maybe_len
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)

from ._network import TransformerPredictionNetwork, TransformerTrainingNetwork
from .trans_decoder import TransformerDecoder
from .trans_encoder import TransformerEncoder


@serde.dataclass
class TransformerEstimator(GluonEstimator):
    """
    Construct a Transformer estimator.

    This implements a Transformer model, close to the one described in
    [Vaswani2017]_.

    .. [Vaswani2017] Vaswani, Ashish, et al. "Attention is all you need."
        Advances in neural information processing systems. 2017.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    trainer
        Trainer object to be used (default: Trainer())
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    cardinality
        Number of values of the each categorical feature (default: [1])
    embedding_dimension
        Dimension of the embeddings for categorical features (the same
        dimension is used for all embeddings, default: 5)
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput())
    model_dim
        Dimension of the transformer network, i.e., embedding dimension of the
        input (default: 32)
    inner_ff_dim_scale
        Dimension scale of the inner hidden layer of the transformer's
        feedforward network (default: 4)
    pre_seq
        Sequence that defined operations of the processing block before the
        main transformer network. Available operations: 'd' for dropout, 'r'
        for residual connections and 'n' for normalization (default: 'dn')
    post_seq
        Sequence that defined operations of the processing block in and after
        the main transformer network. Available operations: 'd' for
        dropout, 'r' for residual connections and 'n' for normalization
        (default: 'drn').
    act_type
        Activation type of the transformer network (default: 'softrelu')
    num_heads
        Number of heads in the multi-head attention (default: 8)
    scaling
        Whether to automatically scale the target values (default: true)
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq)
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference. This is a model optimization that does not affect the
        accuracy (default: 100)
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    batch_size
        The size of the batches to be used training and prediction.
    """

    freq: str
    prediction_length: int = Field(..., gt=0)
    context_length: int = Field(None, gt=0)
    trainer: Trainer = Trainer()
    dropout_rate: float = Field(0.1, ge=0)
    cardinality: List[int] = Field(None, gt=0)
    embedding_dimension: int = Field(20, gt=0)
    distr_output: DistributionOutput = StudentTOutput()
    model_dim: int = 32
    inner_ff_dim_scale: int = 4
    pre_seq: str = "dn"
    post_seq: str = "drn"
    act_type: str = "softrelu"
    num_heads: int = 8
    scaling: bool = True
    lags_seq: List[int] = Field(None)
    time_features: List[TimeFeature] = Field(None)
    use_feat_dynamic_real: bool = False
    use_feat_static_cat: bool = False
    num_parallel_samples: int = Field(100, gt=0)
    train_sampler: InstanceSampler = Field(None)
    validation_sampler: InstanceSampler = Field(None)
    batch_size: int = 32

    def __post_init_post_parse__(self):
        super().__init__(trainer=self.trainer, batch_size=self.batch_size)
        assert (
            self.cardinality is not None or not self.use_feat_static_cat
        ), "You must set `cardinality` if `use_feat_static_cat=True`"

        self.context_length = (
            self.context_length
            if self.context_length is not None
            else self.prediction_length
        )
        self.cardinality = (
            self.cardinality if self.use_feat_static_cat else [1]
        )
        self.lags_seq = (
            self.lags_seq
            if self.lags_seq is not None
            else get_lags_for_frequency(freq_str=self.freq)
        )
        self.time_features = (
            self.time_features
            if self.time_features is not None
            else time_features_from_frequency_str(self.freq)
        )
        self.history_length = self.context_length + max(self.lags_seq)

        self.config = {
            "model_dim": self.model_dim,
            "pre_seq": self.pre_seq,
            "post_seq": self.post_seq,
            "dropout_rate": self.dropout_rate,
            "inner_ff_dim_scale": self.inner_ff_dim_scale,
            "act_type": self.act_type,
            "num_heads": self.num_heads,
        }

        self.encoder = TransformerEncoder(
            self.context_length, self.config, prefix="enc_"
        )
        self.decoder = TransformerDecoder(
            self.prediction_length, self.config, prefix="dec_"
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

    def create_transformation(self) -> Transformation:
        remove_field_names = [
            FieldName.FEAT_DYNAMIC_CAT,
            FieldName.FEAT_STATIC_REAL,
        ]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        empty_list: List[Transformation] = []
        return Chain(
            empty_list
            + [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat
                else []
            )
            + [
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
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
                        if self.use_feat_dynamic_real
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
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            TransformerTrainingNetwork
        )
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
        input_names = get_hybrid_forward_input_names(
            TransformerTrainingNetwork
        )
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> TransformerTrainingNetwork:
        return TransformerTrainingNetwork(
            encoder=self.encoder,
            decoder=self.decoder,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = TransformerPredictionNetwork(
            encoder=self.encoder,
            decoder=self.decoder,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
