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

import itertools
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tsbench.analysis.utils import (
    loocv_split,
    num_fitting_processes,
    run_parallel,
)
from tsbench.config import (
    Config,
    DATASET_REGISTRY,
    DatasetConfig,
    ModelConfig,
    TrainConfig,
)
from tsbench.config.model.models import SeasonalNaiveModelConfig
from tsbench.constants import DEFAULT_DATA_PATH
from tsbench.evaluations.metrics.performance import Metric, Performance
from tsbench.evaluations.tracking import ModelTracker
from tsbench.forecasts import (
    ensemble_forecasts,
    EnsembleWeighting,
    evaluate_forecasts,
    Evaluation,
)


class EnsembleAnalyzer:
    """
    The ensemble analyzer allows for evaluating the performance of ensembles
    across datasets.

    The analysis is run in parallel and should, thus, not be used in a Jupyter
    notebook. Instead, consider using the `tsbench` CLI.
    """

    def __init__(
        self,
        tracker: ModelTracker,
        ensemble_size: Optional[int] = 10,
        ensemble_weighting: EnsembleWeighting = "uniform",
        config_class: Optional[Type[ModelConfig]] = None,
    ):
        """
        Args:
            tracker: The tracker from which to obtain pretrained models and forecasts.
            ensemble_size: The number of models to use when building an ensemble. If not provided,
                uses as many models as possible.
            ensemble_weighting: The type of ensemble weighting to use for averaging forecasts.
            config_class: The class of models to ensemble. If this is provided, fewer models than
                the given ensemble size might be selected.
        """
        self.tracker = tracker
        self.ensemble_size = ensemble_size
        self.ensemble_weighting: EnsembleWeighting = ensemble_weighting
        self.config_class = config_class

    def run(self) -> Tuple[pd.DataFrame, Dict[str, List[ModelConfig]]]:
        """
        Runs the evaluation on the data provided via the tracker. The data
        obtained from the tracker is partitioned by the dataset and we run
        "grouped LOOCV" to compute performance metrics on datasets. Metrics on
        each dataset are then returned as data frame.

        Returns:
            The metrics on the individual datasets.
            The model choices for each dataset.
        """
        results = run_parallel(
            self._run_on_dataset,
            data=list(loocv_split(self.tracker)),
            num_processes=num_fitting_processes(),
        )
        performances = [r[0] for r in results]
        member_mapping = {k: v for r in results for k, v in r[1].items()}

        df = pd.concat(performances).set_index("test_dataset")
        return df, member_mapping

    def _run_on_dataset(
        self,
        data: Tuple[
            Tuple[List[Config[ModelConfig]], List[Performance]],
            Tuple[List[Config[ModelConfig]], List[Performance]],
        ],
    ) -> Tuple[pd.DataFrame, Dict[str, List[ModelConfig]]]:
        # Extract the data
        _, (X_test, y_test) = data

        # Compute the metrics
        performance, members = self._performance_on_dataset(X_test, y_test)

        # Transform into output
        df = Performance.to_dataframe([performance]).assign(
            test_dataset=X_test[0].dataset.name()
        )
        return df, {X_test[0].dataset.name(): members}

    def _performance_on_dataset(
        self,
        X_test: List[Config[ModelConfig]],
        y_test: List[Performance],
    ) -> Tuple[Performance, List[ModelConfig]]:
        # If there are model constraints, we restrict the test set
        if self.config_class is not None:
            indices = [
                i
                for i, c in enumerate(X_test)
                if isinstance(c.model, self.config_class)
                and (
                    not isinstance(c.model, TrainConfig)
                    or c.model.training_fraction == 1
                )
            ]
            X_test = [X_test[i] for i in indices]
            y_test = [y_test[i] for i in indices]

        # After that, we sort by the validation or test nCRPS and pick the ensemble members
        val_scores = [self.tracker.get_validation_scores(x) for x in X_test]
        if any(x is None for x in val_scores):
            order = np.argsort([p.ncrps.mean for p in y_test]).tolist()
        else:
            order = np.argsort([s.ncrps for s in val_scores])  # type: ignore
        choices = order[: (self.ensemble_size or len(order))]

        # In case no model is selected, seasonal naïve is selected (this should always be
        # fast enough to satisfy the latency constraint unless our measurement is faulty or
        # the implementation inefficient)
        if len(choices) == 0:
            choices = [
                i
                for i, c in enumerate(X_test)
                if isinstance(c.model, SeasonalNaiveModelConfig)
            ]

        # Eventually, we get the ensemble members and compute the ensemble performance
        members = [X_test[i].model for i in choices]
        performance = self.get_ensemble_performance(
            members,
            dataset=X_test[0].dataset,
            member_performances=[y_test[i] for i in choices],
        )

        return performance, members

    # ---------------------------------------------------------------------------------------------

    def get_ensemble_performance(
        self,
        models: List[ModelConfig],
        dataset: DatasetConfig,
        member_performances: Optional[List[Performance]] = None,
        num_samples: int = 10,
    ) -> Performance:
        """
        Estimates the performance of a list of models on a particular dataset.
        For this, actually trained models are sampled for each configuration.

        Args:
            models: The list of models to evaluate.
            dataset: The dataset to evaluate on.
            member_performances: The (predicted) performances of the provided models. Used to weigh
                the ensemble members. If not provided, uses the true performances.
            num_samples: The number of samples for estimating the performance.

        Returns:
            The expected performance of the ensemble.
        """
        if member_performances is None:
            member_performances = [
                self.tracker.get_performance(Config(m, dataset))
                for m in models
            ]

        # First, we need to get the forecasts for all models
        forecasts = [
            self.tracker.get_forecasts(Config(m, dataset)) for m in models
        ]

        # Then, we want to construct min(#available_choices, 10) different ensembles by randomly
        # choosing models from the configurations without replacement.
        max_choices = np.prod([len(f) for f in forecasts])
        num_choices = min(max_choices, num_samples)
        pool = itertools.product(*[range(len(f)) for f in forecasts])
        model_combinations = random.sample(list(pool), k=num_choices)

        # Then, we evaluate each of the ensembles
        evaluations = []
        for combination in model_combinations:
            ensembled_forecast = ensemble_forecasts(
                [f[i] for i, f in zip(combination, forecasts)],
                self.ensemble_weighting,
                [p.ncrps.mean for p in member_performances],
            )
            evaluation = evaluate_forecasts(
                ensembled_forecast, dataset.data.test().evaluation()
            )
            evaluations.append(evaluation)

        # And eventually, we build the resulting performance object
        performance = Evaluation.performance(evaluations)
        performance.num_gradient_updates = self._combine_metrics(
            member_performances, lambda p: p.num_gradient_updates
        )
        performance.num_model_parameters = self._combine_metrics(
            member_performances, lambda p: p.num_model_parameters
        )
        performance.latency = self._combine_metrics(
            member_performances, lambda p: p.latency
        )
        performance.training_time = self._combine_metrics(
            member_performances, lambda p: p.training_time
        )
        return performance

    def _combine_metrics(
        self,
        performances: List[Performance],
        metric: Callable[[Performance], Metric],
    ) -> Metric:
        return Metric(
            mean=sum(metric(p).mean for p in performances),
            std=math.sqrt(sum(metric(p).std ** 2 for p in performances)),
        )

    # ---------------------------------------------------------------------------------------------

    def get_all_performances(
        self,
        models: Dict[str, List[ModelConfig]],
        data_path: Path = DEFAULT_DATA_PATH,
    ) -> pd.DataFrame:
        """
        TODO.
        """
        results = []
        for name, dataset in tqdm(DATASET_REGISTRY.items()):
            performance = self.get_ensemble_performance(
                models[name], dataset(data_path)
            )
            df = Performance.to_dataframe([performance]).assign(
                test_dataset=name
            )
            results.append(df)

        return pd.concat(results).set_index("test_dataset")
