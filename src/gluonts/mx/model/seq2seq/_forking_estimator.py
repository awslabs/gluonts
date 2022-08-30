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
from typing import List, Type

import numpy as np
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
from gluonts.model.forecast import Quantile
from gluonts.model.forecast_generator import (
    DistributionForecastGenerator,
    QuantileForecastGenerator,
)
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.block.decoder import Seq2SeqDecoder
from gluonts.mx.block.enc2dec import FutureFeatIntegratorEnc2Dec
from gluonts.mx.block.encoder import Seq2SeqEncoder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.itertools import maybe_len
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddAgeFeature,
    AddConstFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    InstanceSampler,
    RemoveFields,
    RenameFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)

from ._forking_network import (
    ForkingSeq2SeqDistributionPredictionNetwork,
    ForkingSeq2SeqNetworkBase,
    ForkingSeq2SeqPredictionNetwork,
    ForkingSeq2SeqTrainingNetwork,
)
from ._transform import ForkingSequenceSplitter


@serde.dataclass
class ForkingSeq2SeqEstimator(GluonEstimator):
    r"""
    Sequence-to-Sequence (seq2seq) structure with the so-called
    "Forking Sequence" proposed in [WTN+17]_.

    The basic idea is that, given a sequence :math:`x_1, x_2, \cdots, x_T`,
    with a decoding length :math:`\tau`, we learn a NN that solves the
    following series of seq2seq problems:

    .. math::
       :nowrap:

       \begin{eqnarray}
       x_1                     & \mapsto & x_{2:2+\tau}\\
       x_1, x_2                & \mapsto & x_{3:3+\tau}\\
       x_1, x_2, x_3           & \mapsto & x_{4:4+\tau}\\
                               & \ldots  & \\
       x_1, \ldots, x_{T-\tau} & \mapsto & x_{T-\tau+1:T}
       \end{eqnarray}

    Essentially, this means instead of having one cut in the standard seq2seq,
    one has multiple cuts that progress linearly.

    Parameters
    ----------
    encoder
        seq2seq encoder
    decoder
        seq2seq decoder
    quantile_output
        quantile output
    distr_output
        distribution output
    freq
        frequency of the time series.
    prediction_length
        length of the decoding sequence.
    context_length
        length of the encoding sequence. (default: 4 * prediction_length)
    use_past_feat_dynamic_real
        Whether to use the ``past_feat_dynamic_real`` field from the data.
        (default: False)
    use_feat_dynamic_real
        Whether to use the ``feat_dynamic_real`` field from the data.
        (default: False)
    use_feat_static_cat:
        Whether to use the ``feat_static_cat`` field from the data.
        (default: False)
    cardinality: List[int] = None,
        Number of values of each categorical feature.
        This must be set if ``use_feat_static_cat == True``. (default: None)
    embedding_dimension: List[int] = None,
        Dimension of the embeddings for categorical features.
        (default: [min(50, (cat+1)//2) for cat in cardinality])
    add_time_feature
        Adds a set of time features. (default: True)
    add_age_feature
        Adds an age feature. (default: False)
        The age feature starts with a small value at the start of the time
        series and grows over time.
    enable_encoder_dynamic_feature
        Whether the encoder should also be provided with the dynamic features
        (``age``, ``time`` and ``feat_dynamic_real`` if enabled respectively).
        (default: True)
    enable_decoder_dynamic_feature
        Whether the decoder should also be provided with the dynamic features
        (``age``, ``time`` and ``feat_dynamic_real`` if enabled respectively).
        (default: True)
        It makes sense to disable this, if you don't have ``feat_dynamic_real``
        for the prediction range.
    trainer
        trainer (default: Trainer())
    scaling
        Whether to automatically scale the target values. (default: False if
        quantile_output is used, True otherwise)
    scaling_decoder_dynamic_feature
        Whether to automatically scale the dynamic features for the decoder.
        (default: False)
    dtype
        (default: np.float32)
    num_forking
        Decides how much forking to do in the decoder. 1 reduces to seq2seq and
        enc_len reduces to MQ-C(R)NN.
    max_ts_len
        Returns the length of the longest time series in the dataset to be used
        in bounding context_length.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    batch_size
        The size of the batches to be used training and prediction.
    """

    encoder: Seq2SeqEncoder = Field(None)
    decoder: Seq2SeqDecoder = Field(None)
    freq: str = Field(...)
    prediction_length: int = Field(..., gt=0)
    quantile_output: QuantileOutput = Field(None)
    distr_output: DistributionOutput = Field(None)
    context_length: int = Field(None, gt=0)
    use_past_feat_dynamic_real: bool = False
    use_feat_dynamic_real: bool = False
    use_feat_static_cat: bool = False
    cardinality: List[int] = Field(None, gt=0)
    embedding_dimension: List[int] = Field(None, gt=0)
    add_time_feature: bool = True
    add_age_feature: bool = False
    enable_encoder_dynamic_feature: bool = True
    enable_decoder_dynamic_feature: bool = True
    trainer: Trainer = Trainer()
    scaling: bool = Field(None)
    scaling_decoder_dynamic_feature: bool = False
    dtype: Type = np.float32
    num_forking: int = Field(None)
    max_ts_len: int = Field(None)
    train_sampler: InstanceSampler = Field(None)
    validation_sampler: InstanceSampler = Field(None)
    batch_size: int = 32

    def __post_init_post_parse__(self):
        super().__init__(trainer=self.trainer, batch_size=self.batch_size)
        assert (self.distr_output is None) != (self.quantile_output is None)
        assert self.use_feat_static_cat or not self.cardinality, (
            "You should set `cardinality` if and only if"
            " `use_feat_static_cat=True`"
        )

        self.context_length = (
            self.context_length
            if self.context_length is not None
            else 4 * self.prediction_length
        )
        if self.max_ts_len is not None:
            max_pad_len = max(self.max_ts_len - self.prediction_length, 0)
            # Don't allow context_length to be longer than the max pad length
            self.context_length = (
                min(max_pad_len, self.context_length)
                if max_pad_len > 0
                else self.context_length
            )
        self.num_forking = (
            min(self.num_forking, self.context_length)
            if self.num_forking is not None
            else self.context_length
        )

        self.cardinality = (
            self.cardinality
            if self.cardinality and self.use_feat_static_cat
            else [1]
        )
        self.embedding_dimension = (
            self.embedding_dimension
            if self.embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )

        self.use_dynamic_feat = (
            self.use_feat_dynamic_real
            or self.add_age_feature
            or self.add_time_feature
        )

        if self.scaling is None:
            self.scaling = self.quantile_output is None

        self.train_sampler = (
            self.train_sampler
            if self.train_sampler is not None
            else ValidationSplitSampler(min_future=self.prediction_length)
        )
        self.validation_sampler = (
            self.validation_sampler
            if self.validation_sampler is not None
            else ValidationSplitSampler(min_future=self.prediction_length)
        )

    def create_transformation(self) -> Transformation:
        chain: List[Transformation] = []
        dynamic_feat_fields = []
        remove_field_names = [
            FieldName.FEAT_DYNAMIC_CAT,
            FieldName.FEAT_STATIC_REAL,
        ]

        # --- GENERAL TRANSFORMATION CHAIN ---

        # determine unused input
        if not self.use_past_feat_dynamic_real:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if not self.use_feat_static_cat:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)

        chain.extend(
            [
                RemoveFields(field_names=remove_field_names),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
            ]
        )

        # --- TRANSFORMATION CHAIN FOR DYNAMIC FEATURES ---

        if self.add_time_feature:
            chain.append(
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                    dtype=self.dtype,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_TIME)

        if self.add_age_feature:
            chain.append(
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    dtype=self.dtype,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_AGE)

        if self.use_feat_dynamic_real:
            # Backwards compatibility:
            chain.append(
                RenameFields({"dynamic_feat": FieldName.FEAT_DYNAMIC_REAL})
            )
            dynamic_feat_fields.append(FieldName.FEAT_DYNAMIC_REAL)

        # we need to make sure that there is always some dynamic input we will
        # however disregard it in the hybrid forward. the time feature is
        # empty for yearly freq so also adding a dummy feature in the case
        # that the time feature is the only one on
        if len(dynamic_feat_fields) == 0 or (
            not self.add_age_feature
            and not self.use_feat_dynamic_real
            and self.freq == "Y"
        ):
            chain.append(
                AddConstFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_CONST,
                    pred_length=self.prediction_length,
                    # For consistency in case with no dynamic features
                    const=0.0,
                    dtype=self.dtype,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_CONST)

        # now we map all the dynamic input of length context_length +
        # prediction_length onto FieldName.FEAT_DYNAMIC we exclude
        # past_feat_dynamic_real since its length is only context_length
        if len(dynamic_feat_fields) > 1:
            chain.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC,
                    input_fields=dynamic_feat_fields,
                )
            )
        elif len(dynamic_feat_fields) == 1:
            chain.append(
                RenameFields({dynamic_feat_fields[0]: FieldName.FEAT_DYNAMIC})
            )

        # --- TRANSFORMATION CHAIN FOR STATIC FEATURES ---

        if not self.use_feat_static_cat:
            chain.append(
                SetField(
                    output_field=FieldName.FEAT_STATIC_CAT,
                    value=np.array([0], dtype=np.int32),
                )
            )

        return Chain(chain)

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        chain: List[Transformation] = []

        chain.append(
            # because of how the forking decoder works, every time step in
            # context is used for splitting, which is why we use the
            # TestSplitSampler
            ForkingSequenceSplitter(
                instance_sampler=instance_sampler,
                enc_len=self.context_length,
                dec_len=self.prediction_length,
                num_forking=self.num_forking,
                encoder_series_fields=[
                    FieldName.OBSERVED_VALUES,
                    # RTS with past and future values which is never empty
                    # because added dummy constant variable
                    FieldName.FEAT_DYNAMIC,
                ]
                + (
                    # RTS with only past values are only used by the encoder
                    [FieldName.PAST_FEAT_DYNAMIC_REAL]
                    if self.use_past_feat_dynamic_real
                    else []
                ),
                encoder_disabled_fields=(
                    [FieldName.FEAT_DYNAMIC]
                    if not self.enable_encoder_dynamic_feature
                    else []
                )
                + (
                    [FieldName.PAST_FEAT_DYNAMIC_REAL]
                    if not self.enable_encoder_dynamic_feature
                    and self.use_past_feat_dynamic_real
                    else []
                ),
                decoder_series_fields=[
                    # Decoder will use all fields under FEAT_DYNAMIC which are
                    # the RTS with past and future values
                    FieldName.FEAT_DYNAMIC,
                ]
                + ([FieldName.OBSERVED_VALUES] if mode != "test" else []),
                decoder_disabled_fields=(
                    [FieldName.FEAT_DYNAMIC]
                    if not self.enable_decoder_dynamic_feature
                    else []
                ),
                prediction_time_decoder_exclude=[FieldName.OBSERVED_VALUES],
            )
        )

        # past_feat_dynamic features generated above in ForkingSequenceSplitter
        # from those under feat_dynamic - we need to stack with the other
        # short related time series from the system labeled as
        # past_past_feat_dynamic_real. The system labels them as
        # past_feat_dynamic_real and the additional past_ is added to the
        # string in the ForkingSequenceSplitter
        if self.use_past_feat_dynamic_real:
            # Stack features from ForkingSequenceSplitter horizontally since
            # they were transposed so shape is now
            # (enc_len, num_past_feature_dynamic)
            chain.append(
                VstackFeatures(
                    output_field=FieldName.PAST_FEAT_DYNAMIC,
                    input_fields=[
                        "past_" + FieldName.PAST_FEAT_DYNAMIC_REAL,
                        FieldName.PAST_FEAT_DYNAMIC,
                    ],
                    h_stack=True,
                )
            )

        return Chain(chain)

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            ForkingSeq2SeqTrainingNetwork
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
            ForkingSeq2SeqTrainingNetwork
        )
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> ForkingSeq2SeqNetworkBase:
        return ForkingSeq2SeqTrainingNetwork(
            encoder=self.encoder,
            enc2dec=FutureFeatIntegratorEnc2Dec(),
            decoder=self.decoder,
            quantile_output=self.quantile_output,
            distr_output=self.distr_output,
            context_length=self.context_length,
            num_forking=self.num_forking,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            scaling=self.scaling,
            scaling_decoder_dynamic_feature=self.scaling_decoder_dynamic_feature,  # noqa: E501
            dtype=self.dtype,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: ForkingSeq2SeqNetworkBase,
    ) -> Predictor:
        quantile_strs = (
            [
                Quantile.from_float(quantile).name
                for quantile in self.quantile_output.quantiles
            ]
            if self.quantile_output is not None
            else None
        )

        prediction_splitter = self._create_instance_splitter("test")

        prediction_network_class = (
            ForkingSeq2SeqPredictionNetwork
            if self.quantile_output is not None
            else ForkingSeq2SeqDistributionPredictionNetwork
        )

        prediction_network = prediction_network_class(
            encoder=trained_network.encoder,
            enc2dec=trained_network.enc2dec,
            decoder=trained_network.decoder,
            quantile_output=trained_network.quantile_output,
            distr_output=trained_network.distr_output,
            context_length=self.context_length,
            num_forking=self.num_forking,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            scaling=self.scaling,
            scaling_decoder_dynamic_feature=self.scaling_decoder_dynamic_feature,  # noqa: E501
            dtype=self.dtype,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            forecast_generator=(
                QuantileForecastGenerator(quantile_strs)
                if quantile_strs is not None
                else DistributionForecastGenerator(self.distr_output)
            ),
        )
