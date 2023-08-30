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

from abc import ABC, abstractmethod
import torch
from torch import nn
from meta.data.batch import SeriesBatch


class FeatureExtractor(nn.Module, ABC):
    """
    Base class for feature extractors.
    """

    @abstractmethod
    def forward(self, series: SeriesBatch) -> SeriesBatch:
        """
        Computes features for each time point for each time series.

        Parameters
        ----------
        series
            A SeriesBatch containing sequences of size [batch, sequence length, n_input],
            where n_input is typically 1 + * which corresponds to the univariate time series itself
            and additional time features, e.g. relative time.

        Returns
        -------
        SeriesBatch
            The transformed batch with sequences of size [batch, sequence length, n_features].
        """


class IdentityFeatureExtractor(FeatureExtractor):
    """
    Dummy feature extractor which implements the identity function.
    """

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Identity()

    def forward(self, series: SeriesBatch) -> torch.Tensor:
        return self.feature_extractor(series)
