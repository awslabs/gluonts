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
from typing import Any, List, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import TransformerMixin
from tsbench.evaluations.metrics import Performance


class PerformanceTransformer(TransformerMixin):
    """
    The performance transformer transforms performances into model outputs for
    supervised learning as well as model outputs to performance objects.
    """

    def __init__(self, metrics: list[str] | None = None):
        """
        Args:
            metrics: The performance metrics to transform. If `None`, all performance metrics are
                transformed. If provided, metrics that are not present will be converted into NaNs.
        """
        self.encoder = PerformanceEncoder(metrics)

    @property
    def features_names_(self) -> list[str]:
        """
        Returns the feature names for the columns of the transformed
        performance objects.
        """
        return self.encoder.feature_names_

    def fit(self, y: list[Performance]) -> PerformanceTransformer:
        """
        Uses the provided performances to fit the performance transformer.

        Args:
            y: The performance objects.
        """
        self.encoder.fit(y)
        return self

    def transform(self, y: list[Performance]) -> npt.NDArray[np.float32]:
        """
        Transforms the provided performance object into NumPy arrays according
        to the fitted transformer.

        Args:
            y: The performance objects.

        Returns:
            An array of shape [N, K] of transformed performance objects (N: the number of
                performance objects, K: number of performance metrics).
        """
        return self.encoder.transform(y)

    def inverse_transform(
        self, y: npt.NDArray[np.float32]
    ) -> list[Performance]:
        """
        Transforms the provided NumPy arrays back into performance objects
        according to the fitted transformer.

        Args:
            y: A NumPy array of shape [N, K] of performances (N: number of performances, K: number
                of performance metrics).

        Returns:
            The performance objects.
        """
        return self.encoder.inverse_transform(y)


# -------------------------------------------------------------------------------------------------
# pylint: disable=missing-class-docstring,missing-function-docstring


class PerformanceEncoder:
    def __init__(self, metrics: list[str] | None = None):
        self.metrics = metrics
        self.all_feature_names_: list[str]
        self.feature_names_: list[str]

    def fit(self, X: list[Performance], _y: Any = None) -> PerformanceEncoder:
        df = Performance.to_dataframe(X)
        self.all_feature_names_ = df.columns.tolist()
        if self.metrics is None:
            self.feature_names_ = df.columns.tolist()
        else:
            assert all(m in df.columns for m in self.metrics)
            self.feature_names_ = self.metrics
        return self

    def transform(
        self, X: list[Performance], _y: Any = None
    ) -> npt.NDArray[np.float32]:
        df = Performance.to_dataframe(X)
        return df[self.feature_names_].to_numpy()

    def inverse_transform(
        self, X: npt.NDArray[np.float32], _y: Any = None
    ) -> list[Performance]:
        df = pd.DataFrame(X, columns=self.feature_names_).assign(
            **{
                col: np.nan
                for col in set(self.all_feature_names_)
                - set(self.feature_names_)
            }
        )
        return [
            Performance.from_dict(row.to_dict()) for _, row in df.iterrows()
        ]
