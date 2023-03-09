from itertools import chain
from typing import List, Optional, Dict

import numpy as np
import torch

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.model.predictor import Predictor
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.transform import (
    Transformation,
    Chain,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    AddAgeFeature,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetField,
)

from pts import Trainer
from pts.model import PyTorchEstimator
from pts.model.utils import get_module_forward_input_names

from .tft_network import (
    TemporalFusionTransformerPredictionNetwork,
    TemporalFusionTransformerTrainingNetwork,
)
from .tft_transform import BroadcastTo, TFTInstanceSplitter


def _default_feat_args(dims_or_cardinalities: List[int]):
    if dims_or_cardinalities:
        return dims_or_cardinalities
    return [1]


class TemporalFusionTransformerEstimator(PyTorchEstimator):
    @validated()
    def __init__(
            self,
            freq: str,
            prediction_length: int,
            context_length: Optional[int] = None,
            dropout_rate: float = 0.1,
            embed_dim: int = 32,
            num_heads: int = 4,
            num_outputs: int = 3,
            variable_dim: Optional[int] = None,
            time_features: List[TimeFeature] = [],
            static_cardinalities: Dict[str, int] = {},
            dynamic_cardinalities: Dict[str, int] = {},
            static_feature_dims: Dict[str, int] = {},
            dynamic_feature_dims: Dict[str, int] = {},
            past_dynamic_features: List[str] = [],
            trainer: Trainer = Trainer(),
    ) -> None:
        super().__init__(trainer=trainer)

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length

        # MultiheadAttention
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.num_outputs = num_outputs
        self.variable_dim = variable_dim or embed_dim

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
                self.past_dynamic_cardinalities[name] = self.dynamic_cardinalities.pop(
                    name
                )
            elif name in self.dynamic_feature_dims:
                self.past_dynamic_feature_dims[name] = self.dynamic_feature_dims.pop(
                    name
                )
            else:
                raise ValueError(
                    f"Feature name {name} is not provided in feature dicts"
                )

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )

        self.validation_sampler = ValidationSplitSampler(min_future=prediction_length)

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
                    AddAgeFeature(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_AGE,
                        pred_length=self.prediction_length,
                        log_scale=True,
                    ),
                ]
        )

        if self.static_cardinalities:
            transforms.extend([
                VstackFeatures(
                    output_field=FieldName.FEAT_STATIC_CAT,
                    input_fields=list(self.static_cardinalities.keys()),
                    h_stack=True,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=np.long
                )
            ])
        else:
            transforms.extend(
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_CAT,
                        value=[0],
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=np.long
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
                    AsNumpyArray(field=FieldName.FEAT_STATIC_REAL, expected_ndim=1),
                ]
            )

        if self.dynamic_cardinalities:
            transforms.extend([
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_CAT,
                    input_fields=list(self.dynamic_cardinalities.keys()),
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_CAT,
                    expected_ndim=2,
                    dtype=np.long,
                )
            ])
        else:
            transforms.extend(
                [
                    SetField(
                        output_field=FieldName.FEAT_DYNAMIC_CAT,
                        value=[[0]],
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_DYNAMIC_CAT,
                        expected_ndim=2,
                        dtype=np.long,
                    ),
                    BroadcastTo(
                        field=FieldName.FEAT_DYNAMIC_CAT,
                        ext_length=self.prediction_length,
                    ),
                ]
            )

        input_fields = [FieldName.FEAT_TIME, FieldName.FEAT_AGE]
        if self.dynamic_feature_dims:
            input_fields += list(self.dynamic_feature_dims.keys())
        transforms.append(
            VstackFeatures(
                input_fields=input_fields,
                output_field=FieldName.FEAT_DYNAMIC_REAL,
            )
        )

        if self.past_dynamic_cardinalities:
            transforms.extend([
                VstackFeatures(
                    output_field=FieldName.PAST_FEAT_DYNAMIC + "_cat",
                    input_fields=list(self.past_dynamic_cardinalities.keys()),
                ),
                AsNumpyArray(
                    field=FieldName.PAST_FEAT_DYNAMIC + "_cat",
                    expected_ndim=2,
                    dtype=np.long,
                )
            ])
        else:
            transforms.extend(
                [
                    SetField(
                        output_field=FieldName.PAST_FEAT_DYNAMIC + "_cat",
                        value=[[0]],
                    ),
                    AsNumpyArray(
                        field=FieldName.PAST_FEAT_DYNAMIC + "_cat",
                        expected_ndim=2,
                        dtype=np.long,
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

    def create_instance_splitter(self, mode: str):
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

    def create_training_network(
            self, device: torch.device
    ) -> TemporalFusionTransformerTrainingNetwork:
        network = TemporalFusionTransformerTrainingNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            variable_dim=self.variable_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_outputs=self.num_outputs,
            dropout=self.dropout_rate,
            d_past_feat_dynamic_real=_default_feat_args(
                list(self.past_dynamic_feature_dims.values())
            ),
            c_past_feat_dynamic_cat=_default_feat_args(
                list(self.past_dynamic_cardinalities.values())
            ),
            # +1 is for Age Feature
            d_feat_dynamic_real=_default_feat_args(
                [1] * (len(self.time_features) + 1) + list(self.dynamic_feature_dims.values())
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
        )
        return network.to(device)

    def create_predictor(
            self,
            transformation: Transformation,
            trained_network: TemporalFusionTransformerTrainingNetwork,
            device: torch.device,
    ) -> Predictor:

        prediction_network = TemporalFusionTransformerPredictionNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            variable_dim=self.variable_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_outputs=self.num_outputs,
            dropout=self.dropout_rate,
            d_past_feat_dynamic_real=_default_feat_args(
                list(self.past_dynamic_feature_dims.values())
            ),
            c_past_feat_dynamic_cat=_default_feat_args(
                list(self.past_dynamic_cardinalities.values())
            ),
            # +1 is for Age Feature
            d_feat_dynamic_real=_default_feat_args(
                [1] * (len(self.time_features) + 1) + list(self.dynamic_feature_dims.values())
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
        ).to(device)

        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
            forecast_generator=QuantileForecastGenerator(
                quantiles=[str(q) for q in prediction_network.quantiles],
            ),
        )
