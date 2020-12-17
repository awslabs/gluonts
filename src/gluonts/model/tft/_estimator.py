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

from itertools import chain
from typing import Dict, List, Optional

import mxnet as mx
import numpy as np
from mxnet.gluon import HybridBlock

from gluonts.core.component import DType, validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    SetField,
    Transformation,
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


class TemporalFusionTransformerEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        hidden_dim: int = 32,
        variable_dim: Optional[int] = None,
        num_heads: int = 4,
        num_outputs: int = 3,
        num_instance_per_series: int = 100,
        dropout_rate: float = 0.1,
        time_features: List[TimeFeature] = [],
        static_cardinalities: Dict[str, int] = {},
        dynamic_cardinalities: Dict[str, int] = {},
        static_feature_dims: Dict[str, int] = {},
        dynamic_feature_dims: Dict[str, int] = {},
        past_dynamic_features: List[str] = [],
        batch_size: int = 32,
    ) -> None:
        super(TemporalFusionTransformerEstimator, self).__init__(
            trainer=trainer
        )
        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.variable_dim = variable_dim or hidden_dim
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.num_instance_per_series = num_instance_per_series

        if not time_features:
            self.time_features = time_features_from_frequency_str(self.freq)
        else:
            self.time_features = time_features
        self.static_cardinalities = static_cardinalities
        self.dynamic_cardinalities = dynamic_cardinalities
        self.static_feature_dims = static_feature_dims
        self.dynamic_feature_dims = dynamic_feature_dims
        self.past_dynamic_features = past_dynamic_features

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

    def create_transformation(self) -> Transformation:
        transforms = (
            [AsNumpyArray(field=FieldName.TARGET, expected_ndim=1)]
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

        ts_fields = []
        past_ts_fields = []

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
        ts_fields.append(FieldName.FEAT_DYNAMIC_CAT)

        input_fields = [FieldName.FEAT_TIME]
        if self.dynamic_feature_dims:
            input_fields += list(self.dynamic_feature_dims.keys())
        transforms.append(
            VstackFeatures(
                input_fields=input_fields,
                output_field=FieldName.FEAT_DYNAMIC_REAL,
            )
        )
        ts_fields.append(FieldName.FEAT_DYNAMIC_REAL)

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
        past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC + "_cat")

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
        past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)

        transforms.append(
            TFTInstanceSplitter(
                train_sampler=ExpectedNumInstanceSampler(
                    num_instances=self.num_instance_per_series,
                ),
                past_length=self.context_length,
                future_length=self.prediction_length,
                time_series_fields=ts_fields,
                past_time_series_fields=past_ts_fields,
            )
        )

        return Chain(transforms)

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
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            forecast_generator=QuantileForecastGenerator(
                quantiles=[str(q) for q in prediction_network.quantiles],
            ),
        )
