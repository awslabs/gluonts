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
from typing import Generic, TypeVar
from tsbench.config import Config, EnsembleConfig, ModelConfig
from ._evaluations import Evaluations, Performance

T = TypeVar("T", ModelConfig, EnsembleConfig)


class Tracker(ABC, Generic[T]):
    """
    Base class for trackers that draw from local data.
    """

    @abstractmethod
    def get_evaluations(self) -> Evaluations[T]:
        """
        Returns all evaluations from the jobs associated with the tracker.
        """

    @abstractmethod
    def get_performance(self, config: Config[T]) -> Performance:
        """
        Returns the performance metrics for the provided configuration.

        Args:
            config: The configuration object of the type of configuration that the tracker manages.

        Returns:
            The performance metrics.
        """
