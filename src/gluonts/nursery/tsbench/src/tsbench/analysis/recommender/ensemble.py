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

from typing import Dict, List, Tuple
from tsbench.analysis.utils import num_fitting_processes, run_parallel
from tsbench.analysis.utils.loocv import loocv_split
from tsbench.config import Config, EnsembleConfig
from tsbench.evaluations.metrics import Performance
from tsbench.evaluations.tracking import EnsembleTracker
from tsbench.recommender import Recommender


class EnsembleRecommenderAnalyzer:
    """
    The recommender evaluator evaluates the performance of an ensemble
    recommender across datasets.

    The analysis is run in parallel and should, thus, not be used in a Jupyter
    notebook. Instead, consider using the `tsbench` CLI.
    """

    def __init__(
        self,
        tracker: EnsembleTracker,
        recommender: Recommender[EnsembleConfig],
        num_recommendations: int = 10,
    ):
        """
        Args:
            tracker: The tracker from which to obtain data and model performances.
            recommender: The recommender to use for obtaining models.
            num_recommendations: The number of recommendations to perform.
        """
        self.tracker = tracker
        self.recommender = recommender
        self.num_recommendations = num_recommendations

    def run(self) -> List[Dict[str, EnsembleConfig]]:
        """
        Runs the evaluation on all datasets and returns the selected models for
        each dataset. The config evaluator can be used to construct a data
        frame of performances from the configurations.

        Returns:
            The recommended models. The outer list provides the index of the recommendations, i.e.
                the first item of the list provides all the first recommendations of the
                recommender, etc.
        """
        data = list(loocv_split(self.tracker))
        results = run_parallel(
            self._run_on_dataset,
            data=data,
            num_processes=min(
                len(data),
                num_fitting_processes(
                    cpus_per_process=self.recommender.required_cpus,
                    memory_per_process=self.recommender.required_memory,
                ),
            ),
        )

        recommendations = {k: v for r in results for k, v in r.items()}
        return [
            {k: v[i] for k, v in recommendations.items()}
            for i in range(self.num_recommendations)
        ]

    def _run_on_dataset(
        self,
        data: Tuple[
            Tuple[List[Config[EnsembleConfig]], List[Performance]],
            Tuple[List[Config[EnsembleConfig]], List[Performance]],
        ],
    ) -> Dict[str, List[EnsembleConfig]]:
        # Extract the data
        (X_train, y_train), (X_test, _) = data
        dataset = X_test[0].dataset

        # Fit the recommender and predict
        self.recommender.fit(X_train, y_train)
        recommendations = self.recommender.recommend(
            dataset, max_count=self.num_recommendations
        )

        # Return the recommendations
        return {dataset.name(): [r.config for r in recommendations]}
