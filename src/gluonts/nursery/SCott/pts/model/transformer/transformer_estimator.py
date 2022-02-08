from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from pts import Trainer
from pts.dataset import FieldName
from pts.feature import (
    TimeFeature,
    fourier_time_features_from_frequency_str,
    get_fourier_lags_for_frequency,
)
from pts.model import PTSEstimator, Predictor, PTSPredictor, copy_parameters
from pts.modules import DistributionOutput, StudentTOutput
from pts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    RemoveFields,
    AddAgeFeature,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetField,
)
from .transformer_network import (
    TransformerTrainingNetwork,
    TransformerPredictionNetwork,
)


class TransformerEstimator(PTSEstimator):
    def __init__(
        self,
        input_size: int,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        dropout_rate: float = 0.1,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: List[int] = [20],
        distr_output: DistributionOutput = StudentTOutput(),
        d_model: int = 32,
        dim_feedforward_scale: int = 4,
        act_type: str = "gelu",
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        use_feat_static_real: bool = False,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__(trainer=trainer)

        self.input_size = input_size
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.distr_output = distr_output
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real
        self.cardinality = cardinality if use_feat_static_cat else [1]
        self.embedding_dimension = embedding_dimension
        self.num_parallel_samples = num_parallel_samples
        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_fourier_lags_for_frequency(freq_str=freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency_str(self.freq)
        )
        self.history_length = self.context_length + max(self.lags_seq)
        self.scaling = scaling

        self.d_model = d_model
        self.num_heads = num_heads
        self.act_type = act_type
        self.dim_feedforward_scale = dim_feedforward_scale
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

    def create_transformation(self) -> Transformation:
        remove_field_names = [
            FieldName.FEAT_DYNAMIC_CAT,
        ]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.use_feat_static_cat
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.use_feat_static_real
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=np.long,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
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
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.history_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                ),
            ]
        )

    def create_training_network(
        self, device: torch.device
    ) -> TransformerTrainingNetwork:

        training_network = TransformerTrainingNetwork(
            input_size=self.input_size,
            num_heads=self.num_heads,
            act_type=self.act_type,
            dropout_rate=self.dropout_rate,
            d_model=self.d_model,
            dim_feedforward_scale=self.dim_feedforward_scale,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
        ).to(device)

        return training_network

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> Predictor:

        prediction_network = TransformerPredictionNetwork(
            input_size=self.input_size,
            num_heads=self.num_heads,
            act_type=self.act_type,
            dropout_rate=self.dropout_rate,
            d_model=self.d_model,
            dim_feedforward_scale=self.dim_feedforward_scale,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device)

        copy_parameters(trained_network, prediction_network)

        return PTSPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
            output_transform=None,
        )
