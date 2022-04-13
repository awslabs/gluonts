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

from typing import List, Literal, Optional
import numpy as np
import numpy.typing as npt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRanker, XGBRegressor
from tsbench.config import Config, ModelConfig
from tsbench.evaluations.tracking import ModelTracker
from ._base import DatasetFeaturesMixin, OutputNormalization, Surrogate
from ._factory import register_surrogate
from .transformers import ConfigTransformer


@register_surrogate("xgboost")
class XGBoostSurrogate(Surrogate[ModelConfig], DatasetFeaturesMixin):
    """
    The XGBoost surrogate predicts a model's performance on a new dataset by
    using independent XGBoost regressors for each performance metric.

    For this, models and hyperparameters are converted to feature vectors.
    """

    def __init__(
        self,
        tracker: ModelTracker,
        objective: Literal["regression", "ranking"] = "regression",
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
            objective: The optimization objective for the XGBoost estimators.
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

        self.use_ranking = objective == "ranking"
        self.config_transformer = ConfigTransformer(
            add_model_features=True,
            add_dataset_statistics=use_simple_dataset_features,
            add_seasonal_naive_performance=use_seasonal_naive_performance,
            add_catch22_features=use_catch22_features,
            tracker=tracker,
        )

        if self.use_ranking:
            base_estimator = XGBRanker(objective="rank:pairwise", nthread=4)
        else:
            base_estimator = XGBRegressor(nthread=4)
        self.estimator = MultiOutputRegressor(base_estimator)

    @property
    def required_cpus(self) -> int:
        return 4

    def _fit(
        self, X: List[Config[ModelConfig]], y: npt.NDArray[np.float32]
    ) -> None:
        X_numpy = self.config_transformer.fit_transform(X)

        if self.use_ranking:
            # We need to sort by dataset
            encoder = LabelEncoder()
            dataset_indices = encoder.fit_transform(
                [x.dataset.name() for x in X]
            )
            sorting = np.argsort(dataset_indices)

            # Then, sort X and y and assign the group IDs
            X_grouped = [X_numpy[i] for i in sorting]
            y_grouped = [y[i] for i in sorting]
            self.estimator.fit(
                X_grouped, y_grouped, qid=dataset_indices[sorting]
            )
        else:
            self.estimator.fit(X_numpy, y)

    def _predict(
        self, X: List[Config[ModelConfig]]
    ) -> npt.NDArray[np.float32]:
        X_numpy = self.config_transformer.transform(X)
        return self.estimator.predict(X_numpy)
