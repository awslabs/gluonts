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
from ._base import CandidateGenerator, T


class ReplayCandidateGenerator(CandidateGenerator[T]):
    """
    The replay candidate generator simply returns the model configurations seen
    during training.

    If candidates are provided, they are returned as is.
    """

    def __init__(self) -> None:
        """
        Initializes a new replay candidate generator.
        """
        self.cache: List[T] = []

    def fit(self, configs: List[T]) -> None:
        self.cache = configs

    def generate(self, candidates: Optional[List[T]] = None) -> List[T]:
        # Assert trained
        assert self.cache, "Replay candidate generator has not been fitted."

        # Return candidates or cache
        return candidates or self.cache
