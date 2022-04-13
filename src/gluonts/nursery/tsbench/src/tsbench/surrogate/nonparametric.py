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

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import euclidean_distances
from tsbench.config import Config, ModelConfig
from tsbench.config.model.models import SeasonalNaiveModelConfig
from tsbench.evaluations.tracking import ModelTracker
from ._base import DatasetFeaturesMixin, OutputNormalization, Surrogate
from ._factory import register_surrogate
from .transformers import ConfigTransformer


@register_surrogate("nonparametric")
class NonparametricSurrogate(Surrogate[ModelConfig], DatasetFeaturesMixin):
    """
    The nonparametric surrogate predicts a model's performance on a new dataset
    as the average performance across all known datasets.

    Performances are either predicted as ranks or actual values.
    """

    model_performances_: Dict[ModelConfig, npt.NDArray[np.float32]]
    dataset_features_: npt.NDArray[np.float32]

    def __init__(
        self,
        tracker: ModelTracker,
        use_simple_dataset_features: bool = False,
        use_seasonal_naive_performance: bool = False,
        use_catch22_features: bool = False,
        predict: Optional[List[str]] = None,
        output_normalization: OutputNormalization = None,
        impute_simulatable: bool = False,
    ):
        """
        Args:
            tracker: A tracker that can be used to impute latency and number of model parameters
                into model performances. Also, it is required for some input features.
            use_simple_dataset_features: Whether to use dataset features to predict using a
                weighted average.
            use_seasonal_naive_performance: Whether to use the Seasonal Naïve nCRPS as dataset
                featuers. Requires the cacher to be set.
            use_catch22_features: Whether to use catch22 features for datasets statistics. Ignored
                if `use_dataset_features` is not set.
            predict: The metrics to predict. All if not provided.
            output_normalization: The type of normalization to apply to the features of each
                dataset independently. `None` applies no normalization, "quantile" applies quantile
                normalization, and "standard" transforms data to have zero mean and unit variance.
            impute_simulatable: Whether the tracker should impute latency and number of model
                parameters into the returned performance object.
        """
        super().__init__(
            tracker, predict, output_normalization, impute_simulatable
        )

        self.use_dataset_features = any(
            [
                use_simple_dataset_features,
                use_seasonal_naive_performance,
                use_catch22_features,
            ]
        )
        if self.use_dataset_features:
            self.config_transformer = ConfigTransformer(
                add_model_features=False,
                add_dataset_statistics=use_simple_dataset_features,
                add_seasonal_naive_performance=use_seasonal_naive_performance,
                add_catch22_features=use_catch22_features,
                tracker=tracker,
            )

    def _fit(
        self, X: List[Config[ModelConfig]], y: npt.NDArray[np.float32]
    ) -> None:
        # For each model configuration, we store all performances, sorted by dataset
        performances = defaultdict(list)
        datasets = set()
        for xx, yy in zip(X, y):
            datasets.add(xx.dataset)
            performances[xx.model].append(
                {"performance": yy, "dataset": xx.dataset}
            )

        # Then, we assign the model performances and dataset features
        self.model_performances_ = {
            model: np.stack(
                [
                    p["performance"]
                    for p in sorted(
                        data,
                        key=lambda x: x["dataset"].name(),  # type: ignore
                    )
                ]
            )
            for model, data in performances.items()
        }

        # We use the seasonal naive model config here since it is ignored anyway
        if self.use_dataset_features:
            self.dataset_features_ = self.config_transformer.fit_transform(
                [
                    Config(SeasonalNaiveModelConfig(), d)
                    for d in sorted(datasets, key=lambda x: x.name())  # type: ignore
                ]
            )

    def _predict(
        self, X: List[Config[ModelConfig]]
    ) -> npt.NDArray[np.float32]:
        if self.use_dataset_features:
            embeddings = self.config_transformer.transform(X)

        results = []
        for i, x in enumerate(X):
            performance = self.model_performances_[x.model]
            if self.use_dataset_features:
                dataset_embedding = embeddings[i][None, :]  # type: ignore
                # Compute distances
                distances = euclidean_distances(
                    self.dataset_features_, dataset_embedding
                )
                similarity = 1 / distances
                similarity[distances == 0] = float("inf")
                # Compute weighted prediction
                weights = similarity / np.sum(similarity)
                results.append((performance * weights).sum(0))
            else:
                results.append(performance.mean(0))

        return np.stack(results)
