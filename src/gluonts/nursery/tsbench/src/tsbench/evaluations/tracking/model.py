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

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, List, Optional
from tsbench.config import Config, ModelConfig
from tsbench.constants import DEFAULT_DATA_PATH
from tsbench.evaluations import aws
from tsbench.evaluations.metrics import Performance
from tsbench.forecasts.quantile import QuantileForecasts
from ._base import Tracker
from ._evaluations import Evaluations
from ._info import extract_job_infos, ValidationMetric, ValidationScores
from .job import Job, load_jobs_from_analysis, load_jobs_from_directory


class ModelTracker(Tracker[ModelConfig]):
    """
    The tracker may be used to obtain the performance metrics from a set of
    evaluations.
    """

    @classmethod
    def for_experiment(
        cls, name: str, force_refresh: bool = False, **kwargs: Any
    ) -> ModelTracker:
        """
        Loads the data associated with a set of training jobs run on AWS
        Sagemaker. The tracker is cached such that it does not need to re-
        download data in case no new evaluations are available.

        Args:
            name: The name of the experiment.
            force_refresh: Whether to download experiment data even if a tracker is available
                locally. This ensures that new data is fetched.
            kwargs: Additional properties that may be passed to the initializer of the tracker.

        Returns:
            The tracker with all the available data.
        """
        # Generate the filename including all kwargs
        kwargs_suffix = "-".join(
            f"{k}_{v}" for k, v in sorted(kwargs.items(), key=lambda i: i[0])
        )
        if len(kwargs_suffix) > 0:
            kwargs_suffix = f"+{kwargs_suffix}"

        # If available in cache, return
        cache = (
            Path.home()
            / ".cache"
            / "tsbench"
            / f"experiment-{name}{kwargs_suffix}.pickle"
        )
        if cache.exists() and not force_refresh:
            with cache.open("rb") as f:
                return pickle.load(f)

        # Initialize connection to AWS
        analysis = aws.Analysis(name)
        assert all(
            job.status == "Completed" for job in analysis
        ), "Not all jobs have completed."

        # Initialize tracker
        jobs = load_jobs_from_analysis(analysis)
        tracker = ModelTracker(jobs, **kwargs)

        # Cache tracker and return
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb+") as f:
            pickle.dump(tracker, f)
        return tracker

    @classmethod
    def from_directory(cls, directory: Path, **kwargs: Any) -> ModelTracker:
        """
        Loads the data from a set of evaluations stored on file.

        Args:
            directory: The path to the directory which contains the data.
            kwargs: Additional properties that may be passed to the initializer of the tracker.

        Returns:
            The tracker with all the available data.
        """
        # Generate the filename including all kwargs
        kwargs_suffix = "-".join(
            f"{k}_{v}" for k, v in sorted(kwargs.items(), key=lambda i: i[0])
        )
        if len(kwargs_suffix) > 0:
            kwargs_suffix = f"+{kwargs_suffix}"

        # If available in cache, return
        cache = (
            Path.home()
            / ".cache"
            / "tsbench"
            / f"local-{directory.as_posix().replace('/', '-')}{kwargs_suffix}.pickle"
        )
        if cache.exists():
            with cache.open("rb") as f:
                return pickle.load(f)

        # Initialize tracker
        jobs = load_jobs_from_directory(directory)
        tracker = ModelTracker(jobs, **kwargs)

        # Cache tracker and return
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb+") as f:
            pickle.dump(tracker, f)
        return tracker

    # ---------------------------------------------------------------------------------------------

    def __init__(
        self,
        jobs: list[Job],
        validation_metric: ValidationMetric | None = "val_ncrps",
        group_seeds: bool = True,
        data_path: Path = DEFAULT_DATA_PATH,
    ):
        """
        Args:
            jobs: The jobs that the tracker is used for, either obtained from local storage or from
                AWS Sagemaker.
            validation_metric: The metric that should be used to choose models from different
                checkpoints. If set to `None`, models are not loaded from checkpoints but models
                are taken from predefined intervals.
            group_seeds: Whether the same configuration with differing seeds should be grouped.
        """
        self.infos = extract_job_infos(
            jobs,
            validation_metric=validation_metric,
            group_seeds=group_seeds,
            data_path=data_path,
        )
        self.config_map = {info.config: info for info in self.infos}

    def get_evaluations(self) -> Evaluations[ModelConfig]:
        return Evaluations(
            [info.config for info in self.infos],
            [info.performance for info in self.infos],
        )

    def unique_model_configs(self) -> list[ModelConfig]:
        """
        Returns the unique model configurations that are available in the
        experiments managed by this tracker.

        Returns:
            The list of available model configurations.
        """
        return list({c.model for c in self.config_map.keys()})

    def get_training_jobs(self, config: Config[ModelConfig]) -> list[Job]:
        """
        Returns all training jobs associated with the provided configuration.

        Args:
            config: The model and dataset configuration.

        Returns:
            The list of all training jobs.
        """
        return self.config_map[config].jobs

    def get_forecasts(
        self, config: Config[ModelConfig]
    ) -> list[QuantileForecasts]:
        """
        Returns the quantile forecasts of all models associated with the
        provided configuration, i.e. forecasts for the same model trained on
        different seeds.

        Args:
            config: The configuration to obtain forecasts for.

        Returns:
            The list of forecasts for all models.
        """
        info = self.config_map[config]
        result = []
        for i, job in enumerate(info.jobs):
            result.append(job.get_forecast(info.model_indices[i]))
        return result

    def get_validation_scores(
        self, config: Config[ModelConfig]
    ) -> ValidationScores | None:
        """
        Returns the validation scores associated with the provided
        configuration if available.

        Args:
            config: The configuration to query the validation scores for.

        Returns:
            The validation scores. Available if the model associated with the provided
            configuration is trainable (i.e. a deep learning model).
        """
        return self.config_map[config].val_scores

    def get_performance(self, config: Config[ModelConfig]) -> Performance:
        return self.config_map[config].performance

    def __contains__(self, config: Config[ModelConfig]) -> bool:
        return config in self.config_map
