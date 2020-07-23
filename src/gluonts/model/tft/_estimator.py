from typing import Optional, Dict, List
from itertools import chain

import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock

from gluonts.core.component import DType, validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.support.util import copy_parameters
from gluonts.time_feature import (
    TimeFeature,
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
    RemoveFields,
    SetField,
    Transformation,
    VstackFeatures,
)

from ._transform import TFTInstanceSplitter
from ._network import (
    TemporalFusionTransformerTrainingNetwork,
    TemporalFusionTransformerPredictionNetwork,
)


class TemporalFusionTransformerEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        hidden_dim: int = 32,
        variable_dim: Optional[int] = None,
        num_heads: int = 4,
        num_outputs: int = 3,
        num_instance_per_series: int = 100,
        dropout_rate: float = 0.1,
        time_features: Optional[List[TimeFeature]] = None,
        static_cardinalities: Optional[Dict[str, int]] = None,
        dynamic_cardinalities: Optional[Dict[str, int]] = None,
        static_feature_dims: Optional[Dict[str, int]] = None,
        dynamic_feature_dims: Optional[Dict[str, int]] = None,
        past_dynamic_features: Optional[List[str]] = None,
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
        self.context_lenfth = context_length or prediction_length
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.variable_dim = variable_dim or hidden_dim
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.num_instance_per_series = num_instance_per_series

        self.time_features = time_features or time_features_from_frequency_str(
            self.freq
        )
        self.static_cardinalities = static_cardinalities or {}
        self.dynamic_cardinalities = dynamic_cardinalities or {}
        self.static_feature_dims = static_feature_dims or {}
        self.dynamic_feature_dims = dynamic_feature_dims or {}
        self.past_dynamic_features = past_dynamic_features or []

        self.d_past_feat_dynamic_real = []
        self.c_past_feat_dynamic_cat = []
        self.d_feat_dynamic_real = []
        self.c_feat_dynamic_cat = []
        for name, cardinality in self.dynamic_cardinalities.items():
            if name in self.past_dynamic_features:
                self.c_past_feat_dynamic_cat[name] = cardinality
            else:
                self.c_feat_dynamic_cat[name] = cardinality
        for name, dim in self.dynamic_feature_dims.items():
            if name in self.past_dynamic_features:
                self.d_past_feat_dynamic_real[name] = dim
            else:
                self.d_feat_dynamic_real[name] = dim

    def create_transformation(self) -> Transformation:
        transforms = (
            [AsNumpyArray(field=FieldName.TARGET, expected_ndim=1)]
            + [
                AsNumpyArray(field=name, expected_ndim=1)
                for name in chain(
                    self.static_cardinalities.keys(),
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

        time_series_fields = [
            FieldName.FEAT_TIME,
            FieldName.OBSERVED_VALUES,
        ]
        if self.static_cardinalities:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_STATIC_CAT,
                    input_fields=list(self.static_cardinalities.keys()),
                )
            )
        if self.static_feature_dims:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_STATIC_REAL,
                    input_fields=list(self.static_feature_dims.keys()),
                )
            )
        if self.dynamic_cardinalities:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_CAT,
                    input_fields=list(self.dynamic_cardinalities.keys()),
                )
            )
            time_series_fields.append(FieldName.FEAT_DYNAMIC_CAT)
        if self.dynamic_feature_dims:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
                    input_fields=list(self.dynamic_feature_dims.keys()),
                )
            )
            time_series_fields.append(FieldName.FEAT_DYNAMIC_REAL)
        transforms.append(
            TFTInstanceSplitter(
                train_sampler=ExpectedNumInstanceSampler(
                    num_instances=self.num_instance_per_series,
                ),
                past_length=self.history_length,
                future_length=self.prediction_length,
                target_field=FieldName.TARGET,
                time_series_fields=time_series_fields,
                past_time_series_fields=self.past_dynamic_features,
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
            d_past_feat_dynamic_real=self.d_past_feat_dynamic_real,
            c_past_feat_dynamic_cat=self.c_past_feat_dynamic_cat,
            d_feat_dynamic_real=self.d_feat_dynamic_real,
            c_feat_dynamic_cat=self.c_feat_dynamic_cat,
            d_feat_static_real=list(self.static_feature_dims.values()),
            c_feat_static_cat=list(self.static_cardinalities.values()),
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
            d_past_feat_dynamic_real=self.d_past_feat_dynamic_real,
            c_past_feat_dynamic_cat=self.c_past_feat_dynamic_cat,
            d_feat_dynamic_real=self.d_feat_dynamic_real,
            c_feat_dynamic_cat=self.c_feat_dynamic_cat,
            d_feat_static_real=list(self.static_feature_dims.values()),
            c_feat_static_cat=list(self.static_cardinalities.values()),
            dropout=self.dropout_rate,
        )
        copy_parameters(trained_network, prediction_network)
        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
