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
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class CandidateGenerator(ABC, Generic[T]):
    """
    A candidate generator provides candidate model configurations to the
    recommender.
    """

    @abstractmethod
    def fit(self, configs: List[T]) -> None:
        """
        Fits the candidate generator on a list of model configurations seen
        during training.

        Args:
            configs: The model configurations seen during training.
        """

    @abstractmethod
    def generate(self, candidates: Optional[List[T]] = None) -> List[T]:
        """
        Generates a list of possible model configurations according to the
        strategy defined by the class.

        Args:
            candidates: If provided, every model configuration returned must be a member of this
                set.

        Returns:
            The generated model configurations.
        """
