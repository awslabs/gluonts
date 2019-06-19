# Standard library imports
from typing import List, Optional

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts import transform
from gluonts.block.decoder import OneShotDecoder
from gluonts.block.enc2dec import PassThroughEnc2Dec
from gluonts.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    RNNCovariateEncoder,
    MLPEncoder,
    Seq2SeqEncoder,
)
from gluonts.block.feature import FeatureEmbedder
from gluonts.block.quantile_output import QuantileOutput
from gluonts.block.scaler import NOPScaler, Scaler
from gluonts.core.component import validated
from gluonts.model.estimator import GluonEstimator
from gluonts.model.forecast import QuantileForecast, Quantile
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.support.util import copy_parameters
from gluonts.time_feature.lag import time_features_from_frequency_str
from gluonts.trainer import Trainer
from gluonts.transform import ExpectedNumInstanceSampler, FieldName

# Relative imports
from ._seq2seq_network import Seq2SeqPredictionNetwork, Seq2SeqTrainingNetwork


class Seq2SeqEstimator(GluonEstimator):
    """
    Quantile-Regression Sequence-to-Sequence Estimator

    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: List[int],
        embedding_dimension: int,
        encoder: Seq2SeqEncoder,
        decoder_mlp_layer: List[int],
        decoder_mlp_static_dim: int,
        scaler: Scaler = NOPScaler(),
        context_length: Optional[int] = None,
        num_eval_samples: int = 100,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        trainer: Trainer = Trainer(),
    ) -> None:
        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"

        super().__init__(trainer=trainer)

        self.num_eval_samples = num_eval_samples
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.freq = freq
        self.quantiles = quantiles
        self.encoder = encoder
        self.decoder_mlp_layer = decoder_mlp_layer
        self.decoder_mlp_static_dim = decoder_mlp_static_dim
        self.scaler = scaler
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=[embedding_dimension for _ in cardinality],
        )

    def create_transformation(self) -> transform.Transformation:
        return transform.Chain(
            trans=[
                transform.AsNumpyArray(
                    field=FieldName.TARGET, expected_ndim=1
                ),
                transform.AddTimeFeatures(
                    start_field=transform.FieldName.START,
                    target_field=transform.FieldName.TARGET,
                    output_field=transform.FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
                transform.VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                transform.SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                transform.AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT, expected_ndim=1
                ),
                transform.InstanceSplitter(
                    target_field=transform.FieldName.TARGET,
                    is_pad_field=transform.FieldName.IS_PAD,
                    start_field=transform.FieldName.START,
                    forecast_start_field=transform.FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=[FieldName.FEAT_DYNAMIC_REAL],
                ),
            ]
        )

    def create_training_network(self) -> mx.gluon.HybridBlock:
        distribution = QuantileOutput(self.quantiles)

        enc2dec = PassThroughEnc2Dec()
        decoder = OneShotDecoder(
            decoder_length=self.prediction_length,
            layer_sizes=self.decoder_mlp_layer,
            static_outputs_per_time_step=self.decoder_mlp_static_dim,
        )

        training_network = Seq2SeqTrainingNetwork(
            embedder=self.embedder,
            scaler=self.scaler,
            encoder=self.encoder,
            enc2dec=enc2dec,
            decoder=decoder,
            quantile_output=distribution,
        )

        return training_network

    def create_predictor(
        self,
        transformation: transform.Transformation,
        trained_network: Seq2SeqTrainingNetwork,
    ) -> Predictor:
        # todo: this is specific to quantile output
        quantile_strs = [
            Quantile.from_float(quantile).name for quantile in self.quantiles
        ]

        prediction_network = Seq2SeqPredictionNetwork(
            embedder=trained_network.embedder,
            scaler=trained_network.scaler,
            encoder=trained_network.encoder,
            enc2dec=trained_network.enc2dec,
            decoder=trained_network.decoder,
            quantile_output=trained_network.quantile_output,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            forecast_cls_name=QuantileForecast.__name__,
            forecast_kwargs=dict(forecast_keys=quantile_strs),
        )


class MLP2QRForecaster(Seq2SeqEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: List[int],
        embedding_dimension: int,
        encoder_mlp_layer: List[int],
        decoder_mlp_layer: List[int],
        decoder_mlp_static_dim: int,
        scaler: Scaler = NOPScaler,
        context_length: Optional[int] = None,
        num_eval_samples: int = 100,
        quantiles: List[float] = list([0.1, 0.5, 0.9]),
        trainer: Trainer = Trainer(),
    ) -> None:
        encoder = MLPEncoder(layer_sizes=encoder_mlp_layer)
        super(MLP2QRForecaster, self).__init__(
            freq=freq,
            prediction_length=prediction_length,
            encoder=encoder,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            decoder_mlp_layer=decoder_mlp_layer,
            decoder_mlp_static_dim=decoder_mlp_static_dim,
            context_length=context_length,
            scaler=scaler,
            num_eval_samples=num_eval_samples,
            quantiles=quantiles,
            trainer=trainer,
        )


class RNN2QRForecaster(Seq2SeqEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: List[int],
        embedding_dimension: int,
        encoder_rnn_layer: int,
        encoder_rnn_num_hidden: int,
        decoder_mlp_layer: List[int],
        decoder_mlp_static_dim: int,
        encoder_rnn_model: str = 'lstm',
        encoder_rnn_bidirectional: bool = True,
        scaler: Scaler = NOPScaler,
        context_length: Optional[int] = None,
        num_eval_samples: int = 100,
        quantiles: List[float] = list([0.1, 0.5, 0.9]),
        trainer: Trainer = Trainer(),
    ) -> None:
        encoder = RNNCovariateEncoder(
            mode=encoder_rnn_model,
            hidden_size=encoder_rnn_num_hidden,
            num_layers=encoder_rnn_layer,
            bidirectional=encoder_rnn_bidirectional,
        )
        super(RNN2QRForecaster, self).__init__(
            freq=freq,
            prediction_length=prediction_length,
            encoder=encoder,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            decoder_mlp_layer=decoder_mlp_layer,
            decoder_mlp_static_dim=decoder_mlp_static_dim,
            context_length=context_length,
            scaler=scaler,
            num_eval_samples=num_eval_samples,
            quantiles=quantiles,
            trainer=trainer,
        )


class CNN2QRForecaster(Seq2SeqEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: List[int],
        embedding_dimension: int,
        decoder_mlp_layer: List[int],
        decoder_mlp_static_dim: int,
        scaler: Scaler = NOPScaler,
        context_length: Optional[int] = None,
        num_eval_samples: int = 100,
        quantiles: List[float] = list([0.1, 0.5, 0.9]),
        trainer: Trainer = Trainer(),
    ) -> None:
        encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=[1, 3, 9],
            kernel_size_seq=([3] * len([30, 30, 30])),
            channels_seq=[30, 30, 30],
            use_residual=True,
            use_covariates=True,
        )

        super(CNN2QRForecaster, self).__init__(
            freq=freq,
            prediction_length=prediction_length,
            encoder=encoder,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            decoder_mlp_layer=decoder_mlp_layer,
            decoder_mlp_static_dim=decoder_mlp_static_dim,
            context_length=context_length,
            scaler=scaler,
            num_eval_samples=num_eval_samples,
            quantiles=quantiles,
            trainer=trainer,
        )
