# Standard library imports
from typing import List

# First-party imports
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts import transform
from gluonts.block.feature import FeatureEmbedder
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput, StudentTOutput
from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.model.deep_factor.RNNModel import RNNModel
from gluonts.model.deep_factor._network import (
    DeepFactorTrainingNetwork,
    DeepFactorPredictionNetwork,
)
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


# Third-party imports


class DeepFactorEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_layers: int = 1,
        num_hidden: int = 50,
        num_factor: int = 10,
        num_hidden_noise: int = 5,
        num_layers_noise: int = 1,
        cell_type: str = "lstm",
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

        self.global_factor = RNNModel(
            mode=cell_type,
            num_hidden=num_hidden,
            num_layers=num_layers,
            num_output=num_factor,
        )

        self.noise_process = RNNModel(
            mode=cell_type,
            num_hidden=num_hidden_noise,
            num_layers=num_layers_noise,
            num_output=1,
        )

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

    def create_training_network(self) -> DeepFactorTrainingNetwork:
        return DeepFactorTrainingNetwork(
            embedder=FeatureEmbedder(
                cardinalities=self.cardinality,
                embedding_dims=self.embedding_dimensions,
            ),
            global_model=self.global_factor,
            local_model=self.noise_process,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: DeepFactorTrainingNetwork,
    ) -> Predictor:
        prediction_net = DeepFactorPredictionNetwork(
            embedder=trained_network.embedder,
            global_model=trained_network.global_factor,
            local_model=trained_network.local_model,
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


def simple_main():
    import mxnet as mx
    from pprint import pprint

    dataset = get_dataset("electricity", regenerate=False)

    trainer = Trainer(
        ctx=mx.cpu(0),
        epochs=10,
        num_batches_per_epoch=200,
        learning_rate=1e-3,
        hybridize=False,
    )

    cardinality = int(dataset.metadata.feat_static_cat[0].cardinality)
    estimator = DeepFactorEstimator(
        trainer=trainer,
        context_length=168,
        cardinality=[cardinality],
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
    )

    predictor = estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_eval_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    pprint(agg_metrics)


if __name__ == "__main__":
    simple_main()
