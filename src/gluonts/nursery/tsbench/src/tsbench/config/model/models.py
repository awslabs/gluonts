# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast, List, Optional
from gluonts.mx import DeepAREstimator
from gluonts.model.estimator import DummyEstimator, Estimator
from gluonts.mx import NBEATSEstimator
from gluonts.model.naive_2 import Naive2Predictor
from gluonts.model.npts import NPTSEstimator
from gluonts.model.predictor import Predictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.mx import MQCNNEstimator, MQRNNEstimator
from gluonts.mx import SimpleFeedForwardEstimator
from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.trainer.callback import Callback
from gluonts.time_feature import Constant
from mxnet.gluon import nn
from tsbench.config.dataset import DatasetConfig
from tsbench.config.dataset.datasets import WindFarmsDatasetConfig
from ._base import ModelConfig, TrainConfig
from ._factory import register_model


@register_model
@dataclass(frozen=True)
class SeasonalNaiveModelConfig(ModelConfig):
    """
    The seasonal naive estimator.
    """

    @classmethod
    def name(cls) -> str:
        return "seasonal_naive"

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return DummyEstimator(
            SeasonalNaivePredictor,
            freq=freq,
            prediction_length=prediction_length,
        )


@register_model
@dataclass(frozen=True)
class ThetaModelConfig(ModelConfig):
    """
    The Theta R estimator config.
    """

    @classmethod
    def name(cls) -> str:
        return "theta"

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return DummyEstimator(
            RForecastPredictor,
            freq=freq,
            prediction_length=prediction_length,
            method_name="thetaf",
        )


@dataclass(frozen=True)
class Naive2ModelConfig(ModelConfig):
    """
    The Naïve 2 model as used in the M4 competition.
    """

    @classmethod
    def name(cls) -> str:
        return "naive2"

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return DummyEstimator(
            Naive2Predictor, freq=freq, prediction_length=prediction_length
        )


@register_model
@dataclass(frozen=True)
class NptsModelConfig(ModelConfig):
    """
    The NPTS estimator config.
    """

    @classmethod
    def name(cls) -> str:
        return "npts"

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return NPTSEstimator(freq=freq, prediction_length=prediction_length)


@register_model
@dataclass(frozen=True)
class DeepARModelConfig(ModelConfig, TrainConfig):
    """
    The DeepAR estimator config.
    """

    num_layers: int = 2
    num_cells: int = 40

    @classmethod
    def name(cls) -> str:
        return "deepar"

    def create_predictor(
        self, estimator: Estimator, network: nn.HybridBlock
    ) -> Predictor:
        deepar_estimator = cast(DeepAREstimator, estimator)
        transform = deepar_estimator.create_transformation()
        return deepar_estimator.create_predictor(transform, network)

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            time_features=None if time_features else [],
            trainer=self._create_trainer(
                training_time,
                validation_milestones,
                callbacks,  # type: ignore
            ),
            context_length=self.context_length_multiple * prediction_length,
        )


@register_model
@dataclass(frozen=True)
class MQCnnModelConfig(ModelConfig, TrainConfig):
    """
    The MQCNN estimator config.
    """

    context_length_multiple: int = 4
    num_filters: int = 30
    kernel_size_first: int = 7
    kernel_size_hidden: int = 3
    kernel_size_last: int = 3

    @classmethod
    def name(cls) -> str:
        return "mqcnn"

    def create_predictor(
        self, estimator: Estimator, network: nn.HybridBlock
    ) -> Predictor:
        mqcnn_estimator = cast(MQCNNEstimator, estimator)
        transform = mqcnn_estimator.create_transformation()
        return mqcnn_estimator.create_predictor(transform, cast)  # type: ignore

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return MQCNNEstimator(
            freq=freq,
            prediction_length=prediction_length,
            channels_seq=[self.num_filters] * 3,
            kernel_size_seq=[
                self.kernel_size_first,
                self.kernel_size_hidden,
                self.kernel_size_last,
            ],
            add_time_feature=time_features,
            trainer=self._create_trainer(
                training_time,
                validation_milestones,
                callbacks,  # type: ignore
            ),
            context_length=self.context_length_multiple * prediction_length,
        )


@register_model
@dataclass(frozen=True)
class MQRnnModelConfig(ModelConfig, TrainConfig):
    """
    The MQRNN estimator config.
    """

    context_length_multiple: int = 4

    @classmethod
    def name(cls) -> str:
        return "mqrnn"

    def create_predictor(
        self, estimator: Estimator, network: nn.HybridBlock
    ) -> Predictor:
        mqrnn_estimator = cast(MQRNNEstimator, estimator)
        transform = mqrnn_estimator.create_transformation()
        return mqrnn_estimator.create_predictor(transform, network)  # type: ignore

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return MQRNNEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=self._create_trainer(
                training_time,
                validation_milestones,
                callbacks,  # type: ignore
            ),
            context_length=self.context_length_multiple * prediction_length,
        )


@register_model
@dataclass(frozen=True)
class SimpleFeedforwardModelConfig(ModelConfig, TrainConfig):
    """
    The simple feedforward estimator config.
    """

    hidden_dim: int = 40
    num_layers: int = 2

    @classmethod
    def name(cls) -> str:
        return "simple_feedforward"

    def create_predictor(
        self, estimator: Estimator, network: nn.HybridBlock
    ) -> Predictor:
        ff_estimator = cast(SimpleFeedForwardEstimator, estimator)
        transform = ff_estimator.create_transformation()
        return ff_estimator.create_predictor(transform, network)

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return SimpleFeedForwardEstimator(
            freq=freq,
            prediction_length=prediction_length,
            num_hidden_dimensions=[self.hidden_dim] * self.num_layers,
            trainer=self._create_trainer(
                training_time,
                validation_milestones,
                callbacks,  # type: ignore
            ),
            context_length=self.context_length_multiple * prediction_length,
        )


@register_model
@dataclass(frozen=True)
class TemporalFusionTransformerModelConfig(ModelConfig, TrainConfig):
    """
    The temporal fusion transformer estimator config.
    """

    hidden_dim: int = 32
    num_heads: int = 4

    @classmethod
    def name(cls) -> str:
        return "tft"

    def create_predictor(
        self, estimator: Estimator, network: nn.HybridBlock
    ) -> Predictor:
        tft_estimator = cast(TemporalFusionTransformerEstimator, estimator)
        transform = tft_estimator.create_transformation()
        return tft_estimator.create_predictor(transform, network)

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return TemporalFusionTransformerEstimator(
            freq=freq,
            prediction_length=prediction_length,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_outputs=9,
            time_features=[Constant()] if not time_features else [],
            trainer=self._create_trainer(
                training_time,
                validation_milestones,
                callbacks,  # type: ignore
            ),
        )


@register_model
@dataclass(frozen=True)
class NBeatsModelConfig(ModelConfig, TrainConfig):
    """
    The NBeats estimator config.
    """

    context_length_multiple: int = 2
    num_stacks: int = 30
    num_blocks: int = 1

    @classmethod
    def name(cls) -> str:
        return "nbeats"

    @property
    def prediction_samples(self) -> int:
        return 1

    def create_predictor(
        self, estimator: Estimator, network: nn.HybridBlock
    ) -> Predictor:
        nb_estimator = cast(NBEATSEstimator, estimator)
        transform = nb_estimator.create_transformation()
        return nb_estimator.create_predictor(transform, network)

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return NBEATSEstimator(
            freq=freq,
            prediction_length=prediction_length,
            num_stacks=self.num_stacks,
            num_blocks=[self.num_blocks],
            trainer=self._create_trainer(
                training_time,
                validation_milestones,
                callbacks,  # type: ignore
            ),
            context_length=self.context_length_multiple * prediction_length,
        )


@register_model
@dataclass(frozen=True)
class ProphetModelConfig(ModelConfig):
    """
    The Prophet estimator config.
    """

    @classmethod
    def name(cls) -> str:
        return "prophet"

    def save_predictor(self, predictor: Predictor, path: Path) -> None:
        file = path / "metadata.pickle"
        with file.open("w") as f:
            json.dump(
                {
                    "freq": predictor.freq,
                    "prediction_length": predictor.prediction_length,
                },
                f,
            )

    def load_predictor(self, path: Path) -> Predictor:
        file = path / "metadata.pickle"
        with file.open("r") as f:
            meta = json.load(f)
        return ProphetPredictor(
            freq=meta["freq"], prediction_length=meta["prediction_length"]
        )

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return DummyEstimator(
            ProphetPredictor, freq=freq, prediction_length=prediction_length
        )


@register_model
@dataclass(frozen=True)
class STLARModelConfig(ModelConfig):
    """
    The STL-AR estimator config.
    """

    @classmethod
    def name(cls) -> str:
        return "stlar"

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return DummyEstimator(
            RForecastPredictor,
            freq=freq,
            prediction_length=prediction_length,
            method_name="stlar",
        )


@register_model
@dataclass(frozen=True)
class ARIMAModelConfig(ModelConfig):
    """
    The ARIMA estimator config.
    """

    @classmethod
    def name(cls) -> str:
        return "arima"

    @property
    def prefers_parallel_predictions(self) -> bool:
        return True

    def max_time_series_length(self, config: DatasetConfig) -> Optional[int]:
        if isinstance(config, WindFarmsDatasetConfig):
            return 2880
        return 100000

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return DummyEstimator(
            RForecastPredictor,
            freq=freq,
            prediction_length=prediction_length,
            method_name="arima",
        )


@register_model
@dataclass(frozen=True)
class ETSModelConfig(ModelConfig):
    """
    The ETS estimator config.
    """

    @classmethod
    def name(cls) -> str:
        return "ets"

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        return DummyEstimator(
            RForecastPredictor,
            freq=freq,
            prediction_length=prediction_length,
            method_name="ets",
        )
