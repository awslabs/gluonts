from typing import Optional, Dict, List
from itertools import chain

import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock

from gluonts.core.component import DType, validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import RepresentableBlockPredictor
from gluonts.support.util import copy_parameters
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    Transformation,
    VstackFeatures,
)

from ._transform import TFTInstanceSplitter
from ._network import (
    TemporalFusionTransformerTrainingNetwork,
    TemporalFusionTransformerPredictionNetwork,
)
from ._engine import Trainer, QuantileForecastGenerator


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
        self.context_length = context_length or prediction_length
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
        self.static_feature_dims = static_feature_dims or {}
        self.dynamic_cardinalities = dynamic_cardinalities or {}
        self.dynamic_feature_dims = dynamic_feature_dims or {}
        self.past_dynamic_features = past_dynamic_features or []

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

        ts_fields = []
        past_ts_fields = []

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
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC + "cat")

        if self.past_dynamic_feature_dims:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                    input_fields=list(self.past_dynamic_feature_dims.keys()),
                )
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
            d_past_feat_dynamic_real=list(
                self.past_dynamic_feature_dims.values()
            ),
            c_past_feat_dynamic_cat=list(
                self.past_dynamic_cardinalities.values()
            ),
            d_feat_dynamic_real=[1] * len(self.time_features)
            + list(self.dynamic_feature_dims.values()),
            c_feat_dynamic_cat=list(self.dynamic_cardinalities.values()),
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
            d_past_feat_dynamic_real=list(
                self.past_dynamic_feature_dims.values()
            ),
            c_past_feat_dynamic_cat=list(
                self.past_dynamic_cardinalities.values()
            ),
            d_feat_dynamic_real=[1] * len(self.time_features)
            + list(self.dynamic_feature_dims.values()),
            c_feat_dynamic_cat=list(self.dynamic_cardinalities.values()),
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
            forecast_generator=QuantileForecastGenerator(
                quantiles=[str(q) for q in prediction_network.quantiles],
            ),
        )
