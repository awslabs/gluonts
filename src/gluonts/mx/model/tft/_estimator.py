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
from itertools import chain
from typing import Dict, List

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
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.mx.batchify import batchify
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.itertools import maybe_len
from gluonts.time_feature import (
    Constant,
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)

from ._network import (
    TemporalFusionTransformerPredictionNetwork,
    TemporalFusionTransformerTrainingNetwork,
)
from ._transform import BroadcastTo, TFTInstanceSplitter


def _default_feat_args(dims_or_cardinalities: List[int]):
    if dims_or_cardinalities:
        return dims_or_cardinalities
    return [1]


@serde.dataclass
class TemporalFusionTransformerEstimator(GluonEstimator):
    freq: str
    prediction_length: int = Field(..., gt=0)
    context_length: int = Field(None, gt=0)
    trainer: Trainer = Trainer()
    hidden_dim: int = 32
    variable_dim: int = Field(None)
    num_heads: int = 4
    num_outputs: int = 3
    num_instance_per_series: int = 100
    dropout_rate: float = Field(0.1, ge=0)
    time_features: List[TimeFeature] = Field(default_factory=list)
    static_cardinalities: Dict[str, int] = Field(default_factory=dict)
    dynamic_cardinalities: Dict[str, int] = Field(default_factory=dict)
    static_feature_dims: Dict[str, int] = Field(default_factory=dict)
    dynamic_feature_dims: Dict[str, int] = Field(default_factory=dict)
    past_dynamic_features: List[str] = Field(default_factory=list)
    train_sampler: InstanceSampler = Field(None)
    validation_sampler: InstanceSampler = Field(None)
    batch_size: int = 32

    def __post_init_post_parse__(self):
        super().__init__(trainer=self.trainer, batch_size=self.batch_size)

        self.context_length = self.context_length or self.prediction_length
        self.variable_dim = self.variable_dim or self.hidden_dim

        if not self.time_features:
            self.time_features = time_features_from_frequency_str(self.freq)
            if not self.time_features:
                # If time features are empty (as for yearly data), we add a
                # constant feature of 0
                self.time_features = [Constant()]

        self.past_dynamic_cardinalities = {}
        self.past_dynamic_feature_dims = {}
        for name in self.past_dynamic_features:
            if name in self.dynamic_cardinalities:
                self.past_dynamic_cardinalities[
                    name
                ] = self.dynamic_cardinalities.pop(name)
            elif name in self.dynamic_feature_dims:
                self.past_dynamic_feature_dims[
                    name
                ] = self.dynamic_feature_dims.pop(name)
            else:
                raise ValueError(
                    f"Feature name {name} is not provided in feature dicts"
                )

        if self.train_sampler is None:
            self.train_sampler = ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.prediction_length
            )

        if self.validation_sampler is None:
            self.validation_sampler = ValidationSplitSampler(
                min_future=self.prediction_length
            )

    def create_transformation(self) -> Transformation:
        empty_list: List[Transformation] = []
        transforms = (
            empty_list
            + [AsNumpyArray(field=FieldName.TARGET, expected_ndim=1)]
            + (
                [
                    AsNumpyArray(field=name, expected_ndim=1)
                    for name in self.static_cardinalities.keys()
                ]
            )
            + [
                AsNumpyArray(field=name, expected_ndim=1)
                for name in chain(
                    self.static_feature_dims.keys(),
                    self.dynamic_cardinalities.keys(),
                )
            ]
            + [
                AsNumpyArray(field=name, expected_ndim=2)
                for name in self.dynamic_feature_dims.keys()
            ]
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

        if self.static_cardinalities:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_STATIC_CAT,
                    input_fields=list(self.static_cardinalities.keys()),
                    h_stack=True,
                )
            )
        else:
            transforms.extend(
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_CAT,
                        value=[0.0],
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT, expected_ndim=1
                    ),
                ]
            )

        if self.static_feature_dims:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_STATIC_REAL,
                    input_fields=list(self.static_feature_dims.keys()),
                    h_stack=True,
                )
            )
        else:
            transforms.extend(
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL,
                        value=[0.0],
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL, expected_ndim=1
                    ),
                ]
            )

        if self.dynamic_cardinalities:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_CAT,
                    input_fields=list(self.dynamic_cardinalities.keys()),
                )
            )
        else:
            transforms.extend(
                [
                    SetField(
                        output_field=FieldName.FEAT_DYNAMIC_CAT,
                        value=[[0.0]],
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_DYNAMIC_CAT,
                        expected_ndim=2,
                    ),
                    BroadcastTo(
                        field=FieldName.FEAT_DYNAMIC_CAT,
                        ext_length=self.prediction_length,
                    ),
                ]
            )

        input_fields = [FieldName.FEAT_TIME]
        if self.dynamic_feature_dims:
            input_fields += list(self.dynamic_feature_dims.keys())
        transforms.append(
            VstackFeatures(
                input_fields=input_fields,
                output_field=FieldName.FEAT_DYNAMIC_REAL,
            )
        )

        if self.past_dynamic_cardinalities:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.PAST_FEAT_DYNAMIC + "_cat",
                    input_fields=list(self.past_dynamic_cardinalities.keys()),
                )
            )
        else:
            transforms.extend(
                [
                    SetField(
                        output_field=FieldName.PAST_FEAT_DYNAMIC + "_cat",
                        value=[[0.0]],
                    ),
                    AsNumpyArray(
                        field=FieldName.PAST_FEAT_DYNAMIC + "_cat",
                        expected_ndim=2,
                    ),
                    BroadcastTo(field=FieldName.PAST_FEAT_DYNAMIC + "_cat"),
                ]
            )

        if self.past_dynamic_feature_dims:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                    input_fields=list(self.past_dynamic_feature_dims.keys()),
                )
            )
        else:
            transforms.extend(
                [
                    SetField(
                        output_field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                        value=[[0.0]],
                    ),
                    AsNumpyArray(
                        field=FieldName.PAST_FEAT_DYNAMIC_REAL, expected_ndim=2
                    ),
                    BroadcastTo(field=FieldName.PAST_FEAT_DYNAMIC_REAL),
                ]
            )

        return Chain(transforms)

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        ts_fields = [FieldName.FEAT_DYNAMIC_CAT, FieldName.FEAT_DYNAMIC_REAL]
        past_ts_fields = [
            FieldName.PAST_FEAT_DYNAMIC + "_cat",
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
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            TemporalFusionTransformerTrainingNetwork
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
            TemporalFusionTransformerTrainingNetwork
        )
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(
        self,
    ) -> TemporalFusionTransformerTrainingNetwork:
        network = TemporalFusionTransformerTrainingNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_var=self.variable_dim,
            d_hidden=self.hidden_dim,
            n_head=self.num_heads,
            n_output=self.num_outputs,
            d_past_feat_dynamic_real=_default_feat_args(
                list(self.past_dynamic_feature_dims.values())
            ),
            c_past_feat_dynamic_cat=_default_feat_args(
                list(self.past_dynamic_cardinalities.values())
            ),
            d_feat_dynamic_real=_default_feat_args(
                [1] * len(self.time_features)
                + list(self.dynamic_feature_dims.values())
            ),
            c_feat_dynamic_cat=_default_feat_args(
                list(self.dynamic_cardinalities.values())
            ),
            d_feat_static_real=_default_feat_args(
                list(self.static_feature_dims.values()),
            ),
            c_feat_static_cat=_default_feat_args(
                list(self.static_cardinalities.values()),
            ),
            dropout=self.dropout_rate,
        )
        return network

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> RepresentableBlockPredictor:
        prediction_splitter = self._create_instance_splitter("test")
        prediction_network = TemporalFusionTransformerPredictionNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_var=self.variable_dim,
            d_hidden=self.hidden_dim,
            n_head=self.num_heads,
            n_output=self.num_outputs,
            d_past_feat_dynamic_real=_default_feat_args(
                list(self.past_dynamic_feature_dims.values())
            ),
            c_past_feat_dynamic_cat=_default_feat_args(
                list(self.past_dynamic_cardinalities.values())
            ),
            d_feat_dynamic_real=_default_feat_args(
                [1] * len(self.time_features)
                + list(self.dynamic_feature_dims.values())
            ),
            c_feat_dynamic_cat=_default_feat_args(
                list(self.dynamic_cardinalities.values())
            ),
            d_feat_static_real=_default_feat_args(
                list(self.static_feature_dims.values()),
            ),
            c_feat_static_cat=_default_feat_args(
                list(self.static_cardinalities.values()),
            ),
            dropout=self.dropout_rate,
        )
        copy_parameters(trained_network, prediction_network)
        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            forecast_generator=QuantileForecastGenerator(
                quantiles=[
                    str(q) for q in prediction_network.output.quantiles
                ],
            ),
        )
