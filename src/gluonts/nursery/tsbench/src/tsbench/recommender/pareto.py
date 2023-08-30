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
from tsbench.config import Config
from tsbench.evaluations.metrics import Performance
from tsbench.surrogate import Surrogate
from ._base import Recommender, T
from ._factory import register_recommender
from .generator import CandidateGenerator


@register_recommender("pareto")
class ParetoRecommender(Recommender[T]):
    """
    The pareto recommender recommends models by predicting their performance
    using a surrogate model and computing the Pareto front on the predictions.
    """

    def __init__(
        self,
        surrogate: Surrogate[T],
        objectives: List[str],
        focus: Optional[str] = None,
        generator: Optional[CandidateGenerator[T]] = None,
    ):
        """
        Args:
            surrogate: The surrogate model which predicts metrics from models and their
                configurations. The surrogate will be trained when the `fit` method is called.
            objectives: The list of performance metrics to minimize.
            focus: The metric to prefer. Must be either in the list of objectives. If not
                provided, the first metric to optimize is chosen.
            generator: The generator that generates configurations for recommendations. By default,
                this is the replay candidate generator.
        """
        super().__init__(objectives, focus, generator)
        self.surrogate = surrogate

    @property
    def required_cpus(self) -> int:
        return self.surrogate.required_cpus

    @property
    def required_memory(self) -> int:
        return self.surrogate.required_memory

    def fit(
        self, configs: List[Config[T]], performances: List[Performance]
    ) -> None:
        super().fit(configs, performances)
        self.surrogate.fit(configs, performances)

    def _get_performances(self, configs: List[Config[T]]) -> List[Performance]:
        return self.surrogate.predict(configs)
