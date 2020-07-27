from typing import List, Optional

import numpy as np
from mxnet.gluon import HybridBlock
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import RepresentableBlockPredictor
from gluonts.support.util import copy_parameters
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
    InstanceSplitter,
    RemoveFields,
    SetField,
    Transformation,
    VstackFeatures,
    ExpandDimArray,
)

from ._network import (
    SelfAttentionTrainingNetwork,
    SelfAttentionPredictionNetwork,
)
from ._engine import (
    Trainer,
    QuantileForecastGenerator,
)


class SelfAttentionEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinalities: Optional[List[int]] = None,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        model_dim: int = 64,
        ffn_dim_multiplier: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        num_outputs: int = 3,
        kernel_sizes: List[int] = [3, 5, 7, 9],
        distance_encoding: Optional[str] = "dot",
        pre_layer_norm: bool = False,
        dropout: float = 0.1,
        temperature: float = 1.0,
        num_instances_per_series: int = 100,
        time_features: Optional[List[TimeFeature]] = None,
        use_feat_dynamic_real: bool = True,
        use_feat_dynamic_cat: bool = False,
        use_feat_static_real: bool = False,
        use_feat_static_cat: bool = True,
    ):
        super().__init__(trainer=trainer)
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length
        self.model_dim = model_dim
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.cardinalities = cardinalities or []
        self.kernel_sizes = kernel_sizes
        self.distance_encoding = distance_encoding
        self.pre_layer_norm = pre_layer_norm
        self.dropout = dropout
        self.temperature = temperature
        self.num_instances_per_series = num_instances_per_series

        self.time_features = time_features or time_features_from_frequency_str(
            self.freq
        )
        self.use_feat_dynamic_cat = use_feat_dynamic_cat
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real

    def create_transformation(self) -> Transformation:
        transforms = []
        if self.use_feat_dynamic_real:
            transforms.append(
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_REAL, expected_ndim=1,
                )
            )
        if self.use_feat_dynamic_cat:
            transforms.append(
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_CAT, expected_ndim=1,
                )
            )
        if self.use_feat_static_real:
            transforms.append(
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL, expected_ndim=1,
                )
            )
        if self.use_feat_static_cat:
            transforms.append(
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1,)
            )
        time_series_fields = [FieldName.OBSERVED_VALUES]
        if self.use_feat_dynamic_cat:
            time_series_fields.append(FieldName.FEAT_DYNAMIC_CAT)
        if self.use_feat_dynamic_real or (self.time_features is not None):
            time_series_fields.append(FieldName.FEAT_DYNAMIC_REAL)

        transforms.extend(
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
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE,]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(
                        num_instances=self.num_instances_per_series,
                    ),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=time_series_fields,
                    pick_incomplete=True,
                ),
            ]
        )
        return Chain(transforms)

    def create_training_network(self) -> SelfAttentionTrainingNetwork:
        return SelfAttentionTrainingNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_hidden=self.model_dim,
            m_ffn=self.ffn_dim_multiplier,
            n_head=self.num_heads,
            n_layers=self.num_layers,
            n_output=self.num_outputs,
            cardinalities=self.cardinalities,
            kernel_sizes=self.kernel_sizes,
            dist_enc=self.distance_encoding,
            pre_ln=self.pre_layer_norm,
            dropout=self.dropout,
            temperature=self.temperature,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> RepresentableBlockPredictor:
        prediction_network = SelfAttentionPredictionNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_hidden=self.model_dim,
            m_ffn=self.ffn_dim_multiplier,
            n_head=self.num_heads,
            n_layers=self.num_layers,
            n_output=self.num_outputs,
            cardinalities=self.cardinalities,
            kernel_sizes=self.kernel_sizes,
            dist_enc=self.distance_encoding,
            pre_ln=self.pre_layer_norm,
            dropout=self.dropout,
            temperature=self.temperature,
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
