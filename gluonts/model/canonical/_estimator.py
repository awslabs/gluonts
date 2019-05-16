# Standard library imports
from typing import List

# Third-party imports
from mxnet.gluon import HybridBlock, nn

# First-party imports
from gluonts import transform
from gluonts.block.feature import FeatureEmbedder
from gluonts.block.rnn import RNN
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput, StudentTOutput
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.time_feature.lag import time_features_from_frequency_str
from gluonts.trainer import Trainer
from gluonts.transform import (
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    FieldName,
    SetFieldIfNotPresent,
    TestSplitSampler,
    Transformation,
)

# Relative imports
from ._network import CanonicalPredictionNetwork, CanonicalTrainingNetwork


class CanonicalEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        model: HybridBlock,
        is_sequential: bool,
        freq: str,
        context_length: int,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        num_eval_samples: int = 100,
        cardinality: List[int] = list([1]),
        embedding_dimension: int = 10,
        distr_output: DistributionOutput = StudentTOutput(),
    ) -> None:
        super().__init__(trainer=trainer)

        # TODO: error checking
        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.num_sample_paths = num_eval_samples
        self.cardinality = cardinality
        self.embedding_dimensions = [embedding_dimension for _ in cardinality]
        self.model = model
        self.is_sequential = is_sequential

    def create_transformation(self) -> Transformation:
        return Chain(
            trans=[
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
                transform.InstanceSplitter(
                    target_field=transform.FieldName.TARGET,
                    is_pad_field=transform.FieldName.IS_PAD,
                    start_field=transform.FieldName.START,
                    forecast_start_field=transform.FieldName.FORECAST_START,
                    train_sampler=TestSplitSampler(),
                    time_series_fields=[FieldName.FEAT_TIME],
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                ),
            ]
        )

    def create_training_network(self) -> CanonicalTrainingNetwork:
        return CanonicalTrainingNetwork(
            embedder=FeatureEmbedder(
                cardinalities=self.cardinality,
                embedding_dims=self.embedding_dimensions,
            ),
            model=self.model,
            distr_output=self.distr_output,
            is_sequential=self.is_sequential,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: CanonicalTrainingNetwork,
    ) -> Predictor:
        prediction_net = CanonicalPredictionNetwork(
            embedder=trained_network.embedder,
            model=trained_network.model,
            distr_output=trained_network.distr_output,
            is_sequential=trained_network.is_sequential,
            prediction_len=self.prediction_length,
            num_sample_paths=self.num_sample_paths,
            params=trained_network.collect_params(),
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_net,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


class CanonicalRNNEstimator(CanonicalEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        num_layers: int = 1,
        num_cells: int = 50,
        cell_type: str = 'lstm',
        num_eval_samples: int = 100,
        cardinality: List[int] = list([1]),
        embedding_dimension: int = 10,
        distr_output: DistributionOutput = StudentTOutput(),
    ) -> None:
        model = RNN(
            mode=cell_type, num_layers=num_layers, num_hidden=num_cells
        )

        super(CanonicalRNNEstimator, self).__init__(
            model=model,
            is_sequential=True,
            freq=freq,
            context_length=context_length,
            prediction_length=prediction_length,
            trainer=trainer,
            num_eval_samples=num_eval_samples,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            distr_output=distr_output,
        )


class MLPForecasterEstimator(CanonicalEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        hidden_dim_sequence=list([50]),
        num_eval_samples: int = 100,
        cardinality: List[int] = list([1]),
        embedding_dimension: int = 10,
        distr_output: DistributionOutput = StudentTOutput(),
    ) -> None:
        model = nn.HybridSequential()

        for layer, layer_dim in enumerate(hidden_dim_sequence):
            model.add(
                nn.Dense(
                    layer_dim,
                    flatten=False,
                    activation='relu',
                    prefix="mlp_%d_" % layer,
                )
            )

        super(MLPForecasterEstimator, self).__init__(
            model=model,
            is_sequential=False,
            freq=freq,
            context_length=context_length,
            prediction_length=prediction_length,
            trainer=trainer,
            num_eval_samples=num_eval_samples,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            distr_output=distr_output,
        )
