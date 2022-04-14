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

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, List, Literal, Optional, TypeVar
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import (
    LabelEncoder,
    QuantileTransformer,
    StandardScaler,
)
from tsbench.config import Config, EnsembleConfig, ModelConfig
from tsbench.evaluations.metrics import Performance
from tsbench.evaluations.tracking import Tracker
from .transformers import PerformanceTransformer

OutputNormalization = Optional[Literal["quantile", "standard"]]

T = TypeVar("T", ModelConfig, EnsembleConfig)


class Surrogate(ABC, Generic[T]):
    """
    This class defines the interface for any surrogate model which attempts to
    predict performance metrics from model configurations.

    Subclasses may decide to only predict some performance metrics.
    """

    def __init__(
        self,
        tracker: Tracker[T],
        predict: list[str] | None = None,
        output_normalization: OutputNormalization = None,
        impute_simulatable: bool = False,
    ):
        """
        Args:
            tracker: A tracker that can be used to impute latency and number of model parameters
                into model performances. Also, it is required for some input features.
            predict: The metrics to predict. All if not provided.
            output_normalization: The type of normalization to apply to the features of each
                dataset independently. `None` applies no normalization, "quantile" applies quantile
                normalization, and "standard" transforms data to have zero mean and unit variance.
            impute_simulatable: Whether the tracker should impute latency and number of model
                parameters into the returned performance object.
        """
        self.performance_transformer = PerformanceTransformer(metrics=predict)
        self.output_normalization = output_normalization
        self.impute_simulatable = impute_simulatable
        self.tracker = tracker

    @property
    def required_cpus(self) -> int:
        """
        The number of CPUs required for fitting the surrogate.
        """
        return 1

    @property
    def required_memory(self) -> int:
        """
        The amount of memory in GiB required for fitting the surrogate.
        """
        return 1

    def fit(self, X: list[Config[T]], y: list[Performance]) -> None:
        """
        Uses the provided data to fit a model which is able to predict the
        target variables from the input.

        Args:
            X: The input configurations.
            y: The performance values associated with the input configurations.
        """
        y_numpy = self.performance_transformer.fit_transform(y)

        # If we apply any normalization, we do so independently per dataset
        if self.output_normalization is not None:
            # Initialize the transformer
            if self.output_normalization == "quantile":
                transformer = QuantileTransformer()
            else:
                transformer = StandardScaler()

            # Assign indices according to datasets
            encoder = LabelEncoder()
            dataset_indices = encoder.fit_transform(
                [x.dataset.name() for x in X]
            )

            # Then, iterate over datasets and transform the objectives
            result = np.empty_like(y_numpy)
            for i in range(len(encoder.classes_)):
                mask = dataset_indices == i
                result[mask] = transformer.fit_transform(y_numpy[mask])

            # And eventually re-assign the result
            y_numpy = result

        self._fit(X, y_numpy)

    def predict(self, X: list[Config[T]]) -> list[Performance]:
        """
        Predicts the target variables for the given inputs. Typically requires
        `fit` to be called first.

        Args:
            X: The configurations for which to predict performance metrics.

        Returns:
            The predicted performance metrics for the input configurations.
        """
        y = self._predict(X)
        performances = self.performance_transformer.inverse_transform(y)
        if not self.impute_simulatable:
            return performances

        # If the tracker is defined, latency and model parameters are set to the true values as
        # they can be simulated easily.
        for x, predicted_performance in zip(X, performances):
            true_performance = self.tracker.get_performance(x)
            # in-place operations
            predicted_performance.latency = true_performance.latency
            predicted_performance.num_model_parameters = (
                true_performance.num_model_parameters
            )
        return performances

    @abstractmethod
    def _fit(self, X: list[Config[T]], y: npt.NDArray[np.float32]) -> None:
        pass

    @abstractmethod
    def _predict(self, X: list[Config[T]]) -> npt.NDArray[np.float32]:
        pass


class DatasetFeaturesMixin:
    """
    Simple mixin which can be inherited by surrogates to signal that they
    (optionally) use dataset features.
    """
