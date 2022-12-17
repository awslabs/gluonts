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

from abc import ABC
from typing import cast, Generic, List, Optional, TypeVar
from tsbench.config import Config, DatasetConfig, EnsembleConfig, ModelConfig
from tsbench.evaluations.metrics import Performance
from ._recommendation import Recommendation
from .generator import CandidateGenerator, ReplayCandidateGenerator
from .utils import argsort_nondominated

T = TypeVar("T", ModelConfig, EnsembleConfig)


class Recommender(ABC, Generic[T]):
    """
    A recommender uses a surrogate to recommend models and their configurations
    based on desired target metrics.

    This class implements the general interface.
    """

    def __init__(
        self,
        objectives: List[str],
        focus: Optional[str] = None,
        generator: Optional[CandidateGenerator[T]] = None,
    ):
        """
        Args:
            objectives: The list of performance metrics to minimize.
            focus: The metric to prefer. Must be either in the list of objectives. If not
                provided, the first metric to optimize is chosen.
            generator: The generator that generates configurations for recommendations. By default,
                this is the replay candidate generator.
        """
        # Assertions
        assert len(objectives) > 0, "No metrics provided."

        assert (
            focus is None or focus in objectives
        ), "Focus metric not found in metrics to optimize."

        # Initialize attributes
        self.generator = generator or ReplayCandidateGenerator()
        self.objectives = objectives
        self.focus = focus

    @property
    def required_cpus(self) -> int:
        """
        The number of CPUs required for fitting the recommender.
        """
        return 1

    @property
    def required_memory(self) -> int:
        """
        The amount of memory in GiB required for fitting the recommender.
        """
        return 1

    def fit(
        self, configs: List[Config[T]], _performances: List[Performance]
    ) -> None:
        """
        Fits the recommender, including surrogate model and generator, on the
        provided configurations.

        Args:
            configs: The configurations to train on (the generator typically extracts the unique
                model configurations).
            performances: The performances that the surrogate should fit on. The performances must
                align with the provided configurations.
        """
        self.generator.fit(list({c.model for c in configs}))

    def recommend(
        self,
        dataset: DatasetConfig,
        candidates: Optional[List[T]] = None,
        max_count: int = 10,
    ) -> List[Recommendation[T]]:
        """
        This method takes a dataset and a set of constraints and outputs a set
        of recommendations. The recommendations provide both the configurations
        of the recommended model as well as the expected performance.

        Args:
            dataset: The configuration of the dataset for which to recommend a model.
            candidates: A list of model configurations that are allowed to be recommended. If
                `None`, any model configuration is permitted.
            max_count: The maximum number of models to recommend.

        Returns:
            The recommendations which (approximately) satisfy the provided constraints.
        """
        model_configs = self.generator.generate(candidates)
        configs = [Config(m, dataset) for m in model_configs]
        performances = self._get_performances(configs)

        # We construct a data frame, extracting the performance metrics to minimize.
        # Then, we invert the performance metrics for the metrics to maximize.
        df = Performance.to_dataframe(performances)[self.objectives]

        # Then, we perform a nondominated sort
        argsort = argsort_nondominated(
            df.to_numpy(),  # type: ignore
            dim=df.columns.tolist().index(self.focus)
            if self.focus is not None
            else None,
            max_items=max_count,
        )

        # And get the recommendations
        result = []
        for choice in cast(List[int], argsort):
            config = configs[choice]
            recommendation = Recommendation(config.model, performances[choice])
            result.append(recommendation)

        return result

    def _get_performances(self, configs: List[Config[T]]) -> List[Performance]:
        raise NotImplementedError
