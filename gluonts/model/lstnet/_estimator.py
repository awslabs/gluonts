# Standard library imports
from typing import Optional

# Third-party imports
from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.model.estimator import GluonEstimator
from gluonts.model.lstnet._network import LSTNetTrainingNetwork, LSTNetPredictionNetwork
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.trainer import Trainer
from gluonts.transform import (
    Chain,
    ExpectedNumInstanceSampler,
    FieldName,
    InstanceSplitter,
    Transformation,
)


class LSTNetEstimator(GluonEstimator):
    @validated()
    def __init__(
            self,
            skip: int,
            ar_window: int,
            num_series: int,
            freq: str,
            prediction_length: int,
            context_length: Optional[int] = None,
            trainer: Trainer = Trainer(),
            num_eval_samples: int = 100
    ) -> None:
        super().__init__(trainer=trainer)
        self.skip = skip
        self.ar_window = ar_window
        self.num_series = num_series
        self.prediction_length = prediction_length
        self.freq = freq
        self.context_length = context_length
        self.num_samples = num_eval_samples

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=[],  # [FieldName.FEAT_DYNAMIC_REAL]
                    pick_incomplete=True
                )
            ]
        )

    def create_training_network(self) -> HybridBlock:
        # noinspection PyTypeChecker
        return LSTNetTrainingNetwork(
            skip=self.skip,
            ar_window=self.ar_window,
            data_window=self.context_length,
            prediction_length=self.prediction_length,
            num_series=self.num_series,
            num_samples=self.num_samples,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        # noinspection PyTypeChecker
        prediction_network = LSTNetPredictionNetwork(
            skip=self.skip,
            ar_window=self.ar_window,
            data_window=self.context_length,
            prediction_length=self.prediction_length,
            num_series=self.num_series,
            num_samples=self.num_samples,
            params=trained_network.collect_params(),
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
