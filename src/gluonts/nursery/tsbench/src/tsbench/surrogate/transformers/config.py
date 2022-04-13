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
from typing import Any, cast, Dict, List, Optional, Set, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    StandardScaler,
)
from tsbench.config import (
    Config,
    EnsembleConfig,
    MODEL_REGISTRY,
    ModelConfig,
    TrainConfig,
)
from tsbench.config.model.models import SeasonalNaiveModelConfig
from tsbench.evaluations.tracking import ModelTracker


class ConfigTransformer(TransformerMixin):
    """
    The config transformer transforms a configuration (model + dataset) into a
    real-valued vector.
    """

    def __init__(
        self,
        add_model_features: bool = True,
        add_dataset_statistics: bool = True,
        add_seasonal_naive_performance: bool = False,
        add_catch22_features: bool = False,
        tracker: ModelTracker | None = None,
    ):
        """
        Args:
            add_model_type: Whether a one-hot encoding of the model type as well as model
                hyperparameters should be added.
            add_dataset_statistics: Whether simple dataset statistics ought to be added.
            add_seasonal_naive_performance: Whether to add the nCRPS performance of Seasonal Naïve.
                Requires the cacher to be set.
            add_catch_22_features: Whether a dataset's catch22 features ought to be added.
            tracker: An optional tracker to obtain the performance of Seasonal Naïve.
        """
        assert any(
            [
                add_model_features,
                add_dataset_statistics,
                add_seasonal_naive_performance,
                add_catch22_features,
            ]
        ), "ConfigTransformer must be given at least some group of features."
        assert (
            not add_seasonal_naive_performance or tracker is not None
        ), "Tracker must be set if seasonal naive performance is used."

        self.encoders = []
        if add_model_features:
            self.encoders.append(ModelEncoder(transform_full_config=True))
        if add_dataset_statistics:
            self.encoders.append(DatasetStatisticsEncoder())
        if add_seasonal_naive_performance and tracker is not None:
            self.encoders.append(SeasonalNaivePerformanceEncoder(tracker))
        if add_catch22_features:
            self.encoders.append(DatasetCatch22Encoder())

        self.pipeline = make_union(*self.encoders)

    @property
    def feature_names_(self) -> list[str]:
        """
        Returns the feature names for the columns of transformed
        configurations.
        """
        return [
            f
            for e in self.encoders
            for f in e.feature_names_
            if isinstance(e, Encoder)
        ]

    def fit(self, X: list[Config[ModelConfig]]) -> ConfigTransformer:
        """
        Uses the provided configurations to fit the transformer pipeline.

        Args:
            X: The input configurations.
        """
        self.pipeline.fit(X)
        return self

    def transform(
        self, X: list[Config[ModelConfig]]
    ) -> npt.NDArray[np.float32]:
        """
        Transforms the given configurations according to the fitted transformer
        pipeline.

        Args:
            X: The input configurations.

        Returns:
            A NumPy array of shape [N, D]. N is the number of input configurations, D the dimension
                of the vectorized representation.
        """
        return cast(npt.NDArray[np.float32], self.pipeline.transform(X))


class EnsembleConfigTransformer(TransformerMixin):
    """
    The config transformer transforms a set of model configs into a set of
    NumPy arrays.
    """

    model_type_map: dict[str, int]
    attribute_map: dict[str, int]

    def __init__(self):
        self.scaler = StandardScaler()
        self.model_type_map = {}
        self.attribute_map = {}

    @property
    def feature_names_(self) -> list[str]:
        """
        Returns the feature names for the columns of transformed
        configurations.
        """
        return [f"model={m}" for m in self.model_type_map] + list(
            self.attribute_map.keys()
        )

    def fit(
        self, X: list[Config[EnsembleConfig]]
    ) -> EnsembleConfigTransformer:
        """
        Uses the provided configurations to fit the transformer pipeline.

        Args:
            X: The input configurations.
        """
        configs = [model for ensemble in X for model in ensemble.model]

        # Extract model types and attributes
        model_types = {c.name() for c in configs}
        self.model_type_map = {k: i for i, k in enumerate(sorted(model_types))}

        # pylint: disable=no-member
        shared_attributes = set(TrainConfig.__dataclass_fields__.keys())
        attributes = {
            k if k in shared_attributes else f"{c.name()}_{k}"
            for c in configs
            for k in c.__dataclass_fields__.keys()
        }
        self.attribute_map = {
            k: i + len(self.model_type_map)
            for i, k in enumerate(sorted(attributes))
        }

        # Transform the configs to fit the scaler for the attributes
        out = self._transform(configs)
        rhs = out[:, len(MODEL_REGISTRY) :]
        self.scaler.fit(rhs)
        return self

    def transform(
        self, X: list[Config[EnsembleConfig]]
    ) -> list[npt.NDArray[np.float32]]:
        """
        Transforms the given configurations according to the fitted transformer
        pipeline.

        Args:
            X: The input configurations.

        Returns:
            A NumPy array of shape [N, D] for every ensemble configuration where N is the number of
                ensemble members and D the dimensionality. N might differ for list members.
        """
        configs = [model for ensemble in X for model in ensemble.model]
        out = self._transform(configs)
        lhs = out[:, : len(MODEL_REGISTRY)]
        rhs = out[:, len(MODEL_REGISTRY) :]
        out = np.concatenate([lhs, self.scaler.transform(rhs)], axis=1)
        transformed = np.nan_to_num(out, nan=0)
        chunks = np.cumsum([0] + [len(ensemble.model) for ensemble in X])
        return [
            transformed[lower:upper]
            for lower, upper in zip(chunks, chunks[1:])
        ]

    def _transform(
        self, configs: list[ModelConfig]
    ) -> npt.NDArray[np.float32]:
        result = np.empty((len(configs), len(self.feature_names_)))
        result.fill(np.nan)

        # pylint: disable=no-member
        shared_attributes = set(TrainConfig.__dataclass_fields__.keys())
        for i, config in enumerate(configs):
            result[i, self.model_type_map[config.name()]] = 1
            for field in config.__dataclass_fields__:
                if field in shared_attributes:
                    result[i, self.attribute_map[field]] = getattr(
                        config, field
                    )
                else:
                    result[
                        i, self.attribute_map[f"{config.name()}_{field}"]
                    ] = getattr(config, field)

        return result


# -------------------------------------------------------------------------------------------------
# pylint: disable=missing-class-docstring,missing-function-docstring


class Encoder(ABC):
    @property
    @abstractmethod
    def feature_names_(self) -> list[str]:
        pass


class NanImputer:
    def fit(self, _X: list[dict[str, Any]], _y: Any = None) -> NanImputer:
        return self

    def transform(
        self, X: list[dict[str, Any]], _y: Any = None
    ) -> list[dict[str, Any]]:
        df = pd.DataFrame(X)
        return df.to_dict("records")


class Selector:
    def __init__(
        self, use: set[str] | None = None, ignore: set[str] | None = None
    ):
        assert bool(use) != bool(
            ignore
        ), "One of `use` or `ignore` must be set."

        self.use = use or set()
        self.ignore = ignore or set()

    def fit(self, _X: list[dict[str, Any]], _y: Any = None) -> Selector:
        return self

    def transform(
        self, X: list[dict[str, Any]], _y: Any = None
    ) -> list[dict[str, Any]]:
        return [
            {
                k: v
                for k, v in item.items()
                if (not self.use or k in self.use)
                and (not self.ignore or k not in self.ignore)
            }
            for item in X
        ]


# -------------------------------------------------------------------------------------------------


class ModelEncoder(Encoder):
    def __init__(self, transform_full_config: bool = False):
        self.transform_full_config = transform_full_config
        self.model_vectorizer = DictVectorizer(
            dtype=np.float32, sparse=False, sort=True
        )
        self.hp_vectorizer = DictVectorizer(
            dtype=np.float32, sparse=False, sort=True
        )
        self.pipeline = make_pipeline(
            NanImputer(),
            make_union(
                make_pipeline(
                    Selector(use={"model"}),
                    self.model_vectorizer,
                ),
                make_pipeline(
                    Selector(ignore={"model"}),
                    self.hp_vectorizer,
                    StandardScaler(),
                    SimpleImputer(strategy="constant", fill_value=0.0),
                ),
            ),
        )

    @property
    def feature_names_(self) -> list[str]:
        return (
            self.model_vectorizer.feature_names_
            + self.hp_vectorizer.feature_names_
        )

    def fit(
        self,
        X: list[Config[ModelConfig]] | list[ModelConfig],
        _y: Any = None,
    ) -> ModelEncoder:
        if self.transform_full_config:
            self.pipeline.fit(
                [x.model.asdict() for x in cast(List[Config[ModelConfig]], X)]
            )
        else:
            self.pipeline.fit([x.asdict() for x in cast(List[ModelConfig], X)])
        return self

    def transform(
        self,
        X: list[list[Config[ModelConfig]] | list[ModelConfig]],
        _y: Any = None,
    ) -> npt.NDArray[np.float32]:
        if self.transform_full_config:
            return cast(
                npt.NDArray[np.float32],
                self.pipeline.transform(
                    [
                        x.model.asdict()
                        for x in cast(List[Config[ModelConfig]], X)
                    ]
                ),
            )
        return cast(
            npt.NDArray[np.float32],
            self.pipeline.transform(
                [x.asdict() for x in cast(List[ModelConfig], X)]
            ),
        )


class DatasetStatisticsEncoder(Encoder):
    def __init__(self):
        self.unscaled_vectorizer = DictVectorizer(
            dtype=np.float32, sparse=False, sort=True
        )
        self.scaled_vectorizer = DictVectorizer(
            dtype=np.float32, sparse=False, sort=True
        )
        self.pipeline = make_union(
            make_pipeline(
                Selector(use={"integer_dataset", "frequency"}),
                self.unscaled_vectorizer,
            ),
            make_pipeline(
                Selector(ignore={"integer_dataset", "frequency"}),
                self.scaled_vectorizer,
                MinMaxScaler(),  # required for numerical stability
                PowerTransformer(),
            ),
        )

    @property
    def feature_names_(self) -> list[str]:
        return (
            self.unscaled_vectorizer.feature_names_
            + self.scaled_vectorizer.feature_names_
        )

    def fit(
        self, X: list[Config[ModelConfig]], _y: Any = None
    ) -> DatasetStatisticsEncoder:
        self.pipeline.fit(
            [
                {
                    **x.dataset.stats(),
                    "frequency": x.dataset.meta.freq,
                    "prediction_length": x.dataset.meta.prediction_length,
                }
                for x in X
            ]
        )
        return self

    def transform(
        self, X: list[Config[ModelConfig]], _y: Any = None
    ) -> npt.NDArray[np.float32]:
        return cast(
            npt.NDArray[np.float32],
            self.pipeline.transform([x.dataset.stats() for x in X]),
        )


class SeasonalNaivePerformanceEncoder(Encoder):
    def __init__(self, tracker: ModelTracker):
        self.tracker = tracker
        self.scaler = StandardScaler()

    @property
    def feature_names_(self) -> list[str]:
        return ["seasonal_naive_ncrps"]

    def fit(
        self, X: list[Config[ModelConfig]], _y: Any = None
    ) -> SeasonalNaivePerformanceEncoder:
        self.scaler.fit(self._get_performance_array(X))
        return self

    def transform(
        self, X: list[Config[ModelConfig]], _y: Any = None
    ) -> npt.NDArray[np.float32]:
        return cast(
            npt.NDArray[np.float32],
            self.scaler.transform(self._get_performance_array(X)),
        )

    def _get_performance_array(
        self, X: list[Config[ModelConfig]]
    ) -> npt.NDArray[np.float32]:
        return np.array(
            [
                self.tracker.get_performance(
                    Config(SeasonalNaiveModelConfig(), x.dataset)
                ).ncrps.mean
                for x in X
            ]
        )[:, None]


class DatasetCatch22Encoder(Encoder):
    def __init__(self):
        self.vectorizer = DictVectorizer(
            dtype=np.float32, sparse=False, sort=True
        )
        self.pipeline = make_pipeline(
            self.vectorizer,
            PowerTransformer(),
        )

    @property
    def feature_names_(self) -> list[str]:
        return self.vectorizer.feature_names_

    def fit(
        self, X: list[Config[ModelConfig]], _y: Any = None
    ) -> DatasetCatch22Encoder:
        datasets = {x.dataset for x in X}
        features = {d: d.catch22().mean().to_dict() for d in datasets}
        self.pipeline.fit([features[x.dataset] for x in X])
        return self

    def transform(
        self, X: list[Config[ModelConfig]], _y: Any = None
    ) -> npt.NDArray[np.float32]:
        datasets = {x.dataset for x in X}
        features = {d: d.catch22().mean().to_dict() for d in datasets}
        return cast(
            npt.NDArray[np.float32],
            self.pipeline.transform([features[x.dataset] for x in X]),
        )
