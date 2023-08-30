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

from ._base import Recommender
from ._factory import (
    create_ensemble_recommender,
    create_recommender,
    ENSEMBLE_RECOMMENDER_REGISTRY,
    RECOMMENDER_REGISTRY,
)
from .greedy import GreedyRecommender
from .optimal import OptimalRecommender
from .pareto import ParetoRecommender

__all__ = [
    "ENSEMBLE_RECOMMENDER_REGISTRY",
    "GreedyRecommender",
    "OptimalRecommender",
    "ParetoRecommender",
    "RECOMMENDER_REGISTRY",
    "Recommender",
    "create_ensemble_recommender",
    "create_recommender",
]
