from typing import List, Optional

from mxnet.gluon import HybridBlock
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
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
)

from ._network import (
    SelfAttentionTrainingNetwork, 
    SelfAttentionPredictionNetwork,
)


class AsNumpyArrayDefaultNone(AsNumpyArray):
    def transform(self, data: DataEntry) -> DataEntry:
        if self.field not in data.keys():
            data[self.field] = None
            return data
        else:
            return super(AsNumpyArrayDefaultNone, self).transform(data)



class SelfAttentionEstimator(GluonEstimator):
    @validated()
    def __init__(self,
                 freq: str,
                 prediction_length: int,
                 context_length: Optional[int] = None,
                 trainer: Trainer = Trainer(),
                 data_dim: int,
                 model_dim: int,
                 ffn_dim_multiplier: int,
                 num_heads: int,
                 num_layers: int,
                 num_outputs: int,
                 cardinalities: List[int],
                 kernel_sizes: List[int],
                 distance_encoding: Optional[str],
                 pre_layer_norm: bool = False,
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 time_features: Optional[List[TimeFeature]] = None):
        super().__init__(trainer=trainer)
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length
        self.data_dim = data_dim
        self.model_dim = model_dim
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.cardinalities = cardinalities
        self.kernel_sizes = kernel_sizes
        self.distance_encoding = distance_encoding
        self.pre_layer_norm = pre_layer_norm
        self.dropout = dropout
        self.temperature = temperature
        
        self.time_features = time_features or time_features_from_frequency_str(self.freq)
        
    def create_transformation(self) -> Transformation:
        chain = Chain([
            AsNumpyArray(
                field=FieldName.TARGET, 
                expected_ndim=2,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AsNumpyArrayDefaultNone(
                field=FieldName.FEAT_DYNAMIC_CAT,
                expected_ndim=1,
            ),
            AsNumpyArrayDefaultNone(
                field=FieldName.FEAT_DYNAMIC_REAL,
                expected_ndim=2,
            ),
            AsNumpyArrayDefaultNone(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=0,
            ),
            AsNumpyArrayDefaultNone(
                field=FieldName.FEAT_STATIC_REAL,
                expected_ndim=1,
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
                input_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.FEAT_AGE,
                    FieldName.FEAT_DYNAMIC_REAL,
                ],
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                past_length=self.context_length,
                future_length=self.prediction_length,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
        ])
        return chain
    
    def create_training_network(self) -> SelfAttentionTrainingNetwork:
        return SelfAttentionTrainingNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_data=self.data_dim,
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
        
    def create_predictor(self, 
                         transformation: Transformation,
                         trained_network: HybridBlock) -> Predictor:
        prediction_network = SelfAttentionPredictionNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_data=self.data_dim,
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
        )
        