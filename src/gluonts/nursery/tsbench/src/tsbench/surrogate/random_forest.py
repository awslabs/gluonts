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

from typing import List, Optional
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from tsbench.config import Config, ModelConfig
from tsbench.evaluations.tracking import ModelTracker
from tsbench.surrogate._base import DatasetFeaturesMixin
from ._base import DatasetFeaturesMixin, OutputNormalization, Surrogate
from ._factory import register_surrogate
from .transformers import ConfigTransformer


@register_surrogate("randomforest")
class RandomForestSurrogate(Surrogate[ModelConfig], DatasetFeaturesMixin):
    """
    The Random Forest surrogate uses a random forest model to predict model performances.
    """

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

        self.config_transformer = ConfigTransformer(
            add_model_features=True,
            add_dataset_statistics=use_simple_dataset_features,
            add_seasonal_naive_performance=use_seasonal_naive_performance,
            add_catch22_features=use_catch22_features,
            tracker=tracker,
        )

        base_estimator = RandomForestRegressor(n_jobs=1)
        self.estimator = MultiOutputRegressor(base_estimator)

    def _fit(
        self, X: List[Config[ModelConfig]], y: npt.NDArray[np.float32]
    ) -> None:
        X_numpy = self.config_transformer.fit_transform(X)
        self.estimator.fit(X_numpy, y)

    def _predict(
        self, X: List[Config[ModelConfig]]
    ) -> npt.NDArray[np.float32]:
        X_numpy = self.config_transformer.transform(X)
        return self.estimator.predict(X_numpy)
