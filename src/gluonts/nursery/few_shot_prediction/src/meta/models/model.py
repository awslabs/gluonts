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


class MetaModel(ABC, nn.Module):
    """
    Base class for all meta models that make predictions based on a support set and queries
    """

    @abstractmethod
    def forward(self, supps: SeriesBatch, query: SeriesBatch) -> torch.Tensor:
        """
        Computes the forecasts for query past from the support set.

        Parameters
        ----------
        supps
            The support set.
        query
            The queries to be forecasted consisting.

        Returns
        -------
        torch.Tensor
            The forecasts for the entire prediction length with shape `[num_queries,
            prediction_length, *]`.
        """

    @property
    def device(self):
        return next(self.parameters()).device


class SeriesModel(ABC, nn.Module):
    """
    Base class for all models that make predictions based on a query only
    """

    @abstractmethod
    def forward(self, query: SeriesBatch) -> torch.Tensor:
        """
        Computes the forecasts from query past.

        Parameters
        ----------
        query
            The query to be forecasted.

        Returns
        -------
        torch.Tensor
            The forecasts for the entire prediction length with shape `[num_queries,
            prediction_length, *]`.
        """

    @property
    def device(self):
        return next(self.parameters()).device
