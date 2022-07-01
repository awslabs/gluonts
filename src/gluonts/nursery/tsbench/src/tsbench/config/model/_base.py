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

from dataclasses import asdict, dataclass, MISSING
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from mxnet.gluon import nn
from tsbench.config.dataset import DatasetConfig
from tsbench.gluonts import TimedTrainer
from tsbench.gluonts.callbacks import Callback


@dataclass(frozen=True)  # frozen=True is required to make this type hashable
class ModelConfig:
    """
    A model configuration describes the configuration of a GluonTS model and
    allows instantiating an estimator from it.

    Compared to an estimator, this configuration class can easily be hashed and
    different objects with the same configuration are considered equal under
    the `==` operator.
    """

    @classmethod
    def name(cls) -> str:
        """
        Returns the canonical name for the model described by the
        configuration.
        """
        raise NotImplementedError

    @classmethod
    def hyperparameters(cls) -> Dict[str, bool]:
        """
        Returns the list of hyperparameters that are expected by this model
        configuration.

        For each hyperparameter, it provides a boolean whether there exists a
        default value.
        """
        # pylint: disable=no-member
        return {
            key: field.default is not MISSING
            for key, field in cls.__dataclass_fields__.items()  # type: ignore
        }

    def save_predictor(self, predictor: Predictor, path: Path) -> None:
        """
        Saves the predictor associated with the model configuration to the
        specified path. By default, this simply serializes the predictor.

        Args:
            predictor: The predictor to save.
            path: The directory where to save the predictor.
        """
        predictor.serialize(path)

    def load_predictor(self, path: Path) -> Predictor:
        """
        Loads the predictor from the specified path.

        Args:
            path: The directory from which to load the predictor.

        Returns:
            The predictor which was loaded.
        """
        return Predictor.deserialize(path)

    @property
    def prediction_samples(self) -> int:
        """
        The number of samples which should be produced when sampling during
        inference.
        """
        return 100

    @property
    def prefers_parallel_predictions(self) -> bool:
        """
        Returns whether predictions should be generated in parallel.
        """
        return False

    def max_time_series_length(self, _config: DatasetConfig) -> Optional[int]:
        """
        The maximum length a time series may have to be used for prediction. If
        an integer N is provided, only the most recent N observations are used.

        Args:
            config: The dataset for which to determine the maximum time series length.
        """
        return None

    def create_estimator(
        self,
        freq: str,
        prediction_length: int,
        time_features: bool,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> Estimator:
        """
        Creates a GluonTS estimator from the model's configuration.

        Args:
            freq: The frequency of the data that the estimator is created for.
            prediction_length: The required length of the model predictions.
            training_time: The number of seconds to train for.
            validation_milestones: The milestones used for running validation.
            callbacks: An optional list of callbacks to use during training.

        Returns:
            An initialized GluonTS estimator.
        """
        raise NotImplementedError

    def asdict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the model configuration.
        """
        return {
            "model": self.name(),
            **{
                (
                    k
                    if isinstance(self, TrainConfig)
                    and k in TrainConfig.training_hyperparameters()
                    else f"{self.name()}_{k}"
                ): v
                for k, v in asdict(self).items()
            },
        }


# -------------------------------------------------------------------------------------------------
C = TypeVar("C", bound="TrainConfig")


@dataclass(frozen=True)
class TrainConfig:
    """
    The training configuration may be derived by any estimator which is trained
    via the GluonTS `Trainer` class.
    """

    @classmethod
    def training_hyperparameters(cls) -> List[str]:
        """
        Returns the list of hyperparameters that are used for trainable models.

        Returns:
            The list of training hyperparameters.
        """
        # pylint: disable=no-member
        return list(cls.__dataclass_fields__.keys())  # type: ignore

    training_fraction: float = 1.0
    learning_rate: float = 1e-3
    context_length_multiple: int = 1

    def create_predictor(
        self, estimator: Estimator, network: nn.HybridBlock
    ) -> Predictor:
        """
        Creates a predictor from the provided network. This method is required
        to be implemented by every model configuration which describes a
        trainable model.

        Args:
            estimator: The estimator for which to obtain the predictor. Must have been obtained
            from the same model configuration. Otherwise, behavior is undefined.
            network: The network which has been trained.

        Returns:
            The predictor created from the network.
        """
        raise NotImplementedError

    def _create_trainer(
        self,
        training_time: float,
        validation_milestones: List[float],
        callbacks: List[Callback],
    ) -> TimedTrainer:
        return TimedTrainer(
            training_time=training_time,
            validation_milestones=validation_milestones,
            learning_rate=self.learning_rate,
            callbacks=callbacks,
        )
