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

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import pygmo
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tsbench.config import Config, DatasetConfig, ModelConfig
from tsbench.evaluations.metrics import Performance
from ._base import Recommendation, Recommender
from ._factory import register_recommender
from .generator import CandidateGenerator


@register_recommender("greedy")
class GreedyRecommender(Recommender[ModelConfig]):
    """
    Recommender that selects a set of configurations differently in the one-
    dimensional and two- dimensional setting.

    In the one-dimensional setting, it greedily picks the configurations which
    provide the lowest joint error. In the multi-dimensional setting, it
    greedily picks the configurations which provide the lowest joint
    hypervolume error.
    """

    metrics: np.ndarray  # shape [num_datasets, num_models, num_objectives]
    model_indices: Dict[
        ModelConfig, int
    ]  # A map from model config to index in the metrics

    def __init__(
        self,
        objectives: List[str],
        focus: Optional[str] = None,
        generator: Optional[CandidateGenerator[ModelConfig]] = None,
        enforce_single_objective: bool = False,
    ):
        """
        Args:
            objectives: The list of performance metrics to minimize.
            focus: The metric to prefer. Must be either in the list of objectives. If not
                provided, the first metric to optimize is chosen.
            generator: The generator that generates configurations for recommendations. By default,
                this is the replay candidate generator.
            enforce_single_objective: Whether the greedy recommender should pick models greedily
                even in the multi-objective case and pick models alternately.
        """
        super().__init__(objectives, focus, generator)
        self.enforce_single_objective = enforce_single_objective

    def fit(
        self,
        configs: List[Config[ModelConfig]],
        performances: List[Performance],
    ) -> None:
        super().fit(configs, performances)

        # We need to sort by dataset to have the same ordering for each model config
        ordering = np.argsort([c.dataset.name() for c in configs])
        performance_df = Performance.to_dataframe(performances)

        # Extract all metrics
        metric_map = defaultdict(list)
        for i in ordering:
            metric_map[configs[i].model].append(
                performance_df.iloc[i][self.objectives].to_numpy(),  # type: ignore
            )

        # Build the properties
        self.metrics = np.stack(list(metric_map.values()), axis=1)
        self.model_indices = {model: i for i, model in enumerate(metric_map)}

        # If we are in the multi-objective setting, we have to apply dataset-level quantile
        # normalization of each objective. Otherwise, we perform standardization.
        if not self.enforce_single_objective and len(self.objectives) > 1:
            transformer = QuantileTransformer(
                n_quantiles=min(1000, self.metrics.shape[0])
            )
            self.metrics = np.stack(
                [
                    transformer.fit_transform(dataset_metrics)
                    for dataset_metrics in self.metrics
                ]
            )
        else:
            transformer = StandardScaler()
            self.metrics = np.stack(
                [
                    transformer.fit_transform(dataset_metrics)
                    for dataset_metrics in self.metrics
                ]
            )

    def recommend(
        self,
        dataset: DatasetConfig,
        candidates: Optional[List[ModelConfig]] = None,
        max_count: int = 10,
    ) -> List[Recommendation[ModelConfig]]:
        # Since recommendations are independent of the dataset, we could already compute the
        # result of this function in `fit`. However, we don't have access to `max_count` there.
        # First, we get the model configurations and check that they overlap with the ones we know.
        model_configs = self.generator.generate(candidates)
        assert all(c in self.model_indices for c in model_configs), (
            "Greedy recommender can only provide recommendations for known"
            " configurations."
        )

        # Greedily pick configurations. For the multi-objective case, we need to compute the
        # hypervolume error. Hence, for each dataset, we need to compute the hypervolume of the
        # true Pareto front.
        if not self.enforce_single_objective and len(self.objectives) > 1:
            reference = np.ones(len(self.objectives))
            hypervolumes = np.array(
                [
                    pygmo.hypervolume(dataset_metrics).compute(reference)  # type: ignore
                    for dataset_metrics in self.metrics
                ]
            )

        available_choices = list(range(len(model_configs)))
        result = []
        while len(result) < max_count:
            # Pick the configuration which minimizes the joint error. For this, we need to compute
            # the error for all possible choices.
            errors = []
            for i, choice in enumerate(available_choices):
                # Compute the joint error with the model configuration of the choice
                all_configs = result + [model_configs[choice]]
                all_performances = self.metrics[
                    :, [self.model_indices[c] for c in all_configs]
                ]

                if self.enforce_single_objective or len(self.objectives) == 1:
                    # If we only consider a single objective, we can simply sum together the
                    # minimum observed values of the objective across the dataset
                    error = all_performances.min(1)[
                        :, i % len(self.objectives)
                    ].sum()
                else:
                    # Otherwise, we need to compute the hypervolumes for all datasets
                    reference = np.ones(len(self.objectives))
                    config_hypervolumes = np.array(
                        [
                            pygmo.hypervolume(  # type: ignore
                                dataset_performances,
                            ).compute(reference)
                            for dataset_performances in all_performances
                        ]
                    )
                    # And then compute the cumulative hypervolume error
                    error = (hypervolumes - config_hypervolumes).sum()  # type: ignore

                errors.append(error)

            # Get the index with the lowest loss
            lowest = np.argmin(errors)
            index = available_choices[lowest]
            del available_choices[lowest]
            result.append(model_configs[index])

        # Eventually return the configurations. The performance cannot be predicted by the greedy
        # surrogate.
        return [Recommendation(r, _dummy_performance()) for r in result]


def _dummy_performance() -> Performance:
    return Performance.from_dict(
        {
            mm: np.nan
            for m in Performance.metrics()
            for mm in [f"{m}_mean", f"{m}_std"]
        }
    )
