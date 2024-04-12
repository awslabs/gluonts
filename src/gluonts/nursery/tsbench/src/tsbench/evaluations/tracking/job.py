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
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from tsbench.config import MODEL_REGISTRY
from tsbench.evaluations.aws import Analysis, TrainingJob
from tsbench.evaluations.metrics import Metric, Performance
from tsbench.forecasts import QuantileForecasts


class Job:
    """
    A job provides data available for a single training job.

    It is first obtained from a training job run on AWS Sagemaker. By storing
    it locally, it can be loaded from the file system at a later point and no
    Sagemaker connection is required. The initializer of this class should not
    be used directly.
    """

    @classmethod
    def from_training_job(cls, job: TrainingJob) -> Job:
        """
        Initializes a new job from an AWS Sagemaker training job.
        """
        return Job(
            job.hyperparameters["model"],
            job.hyperparameters["dataset"],
            _extract_configuration(job),
            _extract_performance(job),
            source_job=job,
        )

    @classmethod
    def load(cls, directory: Path) -> Job:
        """
        Loads the job from the file system at the specified directory. This
        should only be called on a directory where a job has previously been
        saved.

        Args:
            directory: The directory where to load the job from.

        Returns:
            The training job loaded from the provided path.
        """
        model = directory.parts[-3]
        dataset = directory.parts[-2]

        with (directory / "config.json").open("r") as f:
            config = json.load(f)
        with (directory / "performance.json").open("r") as f:
            performance = json.load(f)

        return Job(model, dataset, config, performance, source_path=directory)

    def __init__(
        self,
        model: str,
        dataset: str,
        config: dict[str, Any],
        performance: dict[str, Any],
        source_path: Path | None = None,
        source_job: TrainingJob | None = None,
    ):
        """
        Args:
            model: The model that the experiment was run for.
            dataset: The dataset that the model was trained on.
            config: The configuration as a dictionary. Must contain the seed and optionally some
                hyperparameters for the model.
            performance: The performances of all models associated with the training job.
            source_path: An optional path in the file system where the job was loaded from. This is
                required to load forecasts from the file. Mutually exclusive with `source_job`.
            source_job: An optional Sagemaker training job that this job was initialized from.
                Required to call the `save` method. Mutually exclusive with `source_path`.
        """
        assert (source_path is None) != (
            source_job is None
        ), "Exactly one of `source_path` or `source_job` must bet set."

        self.model = model
        self.dataset = dataset
        self.config = config
        self.performance = performance
        self.source_path = source_path
        self.source_job = source_job

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """
        Returns the hyperparameters associated with the job.
        """
        return self.config["hyperparameters"]

    @property
    def static_metrics(self) -> dict[str, float | int]:
        """
        Returns job metrics for which there exists only a single value.
        """
        return self.performance["meta"]

    @property
    def metrics(self) -> list[dict[str, dict[str, float | int]]]:
        """
        Returns job metrics that might be available for several models (i.e.
        checkpoints).

        The returned list provides the metrics, ordered by model.
        """
        return self.performance["performances"]

    @property
    def performances(self) -> list[Performance]:
        """
        Returns the list of performances for all models associated with this
        job.

        The variances of all metrics will be set to 0.
        """
        return [
            Performance(
                training_time=Metric(p["training"]["duration"], 0),
                latency=Metric(self.static_metrics["latency"], 0),
                num_model_parameters=Metric(
                    self.static_metrics["num_model_parameters"], 0
                ),
                num_gradient_updates=Metric(
                    p["training"]["num_gradient_updates"], 0
                ),
                **{
                    k: Metric(p["testing"][k], 0)
                    for k in ["mase", "smape", "nrmse", "nd", "ncrps"]
                },
            )
            for p in self.metrics
        ]

    def get_forecast(self, index: int) -> QuantileForecasts:
        """
        Loads the forecasts on the test set for the model with the specified
        index.

        Args:
            index: The index of the model among all models trained.

        Returns:
            The quantile forecasts on the test set.
        """
        # If this job was initialized from Sagemaker, load it from the Sagemaker job
        if self.source_job is not None:
            with self.source_job.artifact(cache=False) as artifact:
                return QuantileForecasts.load(
                    artifact.path / "predictions" / f"model_{index}"
                )

        # Otherwise, load it from the file system
        return QuantileForecasts.load(
            cast(Path, self.source_path) / "forecasts" / f"model_{index:02}"
        )

    def save(self, path: Path, include_forecasts: bool = True) -> None:
        """
        Stores all data associated with the training job in an auto-generated,
        unique folder within the provided directory.

        The job is stored under `<model>/<dataset>/<autogenerated>`. Storing
        jobs via this function allows to decouple experimental results
        completely from AWS Sagemaker.
        """
        assert self.source_job is not None, (
            "Job cannot be saved if it was not initialized from an AWS"
            " Sagemaker job."
        )

        # First, we generate the folder name
        components = [f"seed-{self.config['seed']}"] + [
            f"{k}-{v}"
            for k, v in self.config.get("hyperparameters", {}).items()
        ]
        target = (
            path / self.model / self.dataset / "+".join(sorted(components))
        )

        # Make sure folder exists and is empty
        if target.exists():
            if _check_all_data_available(target):
                return
            shutil.rmtree(target)

        target.mkdir(parents=True, exist_ok=True)

        # Then, we store metadata files
        with (target / "config.json").open("w+") as f:
            json.dump(self.config, f, indent=4)
        with (target / "performance.json").open("w+") as f:
            json.dump(self.performance, f, indent=4)

        # As well as the forecasts (we ignore val forecasts as they are never used)
        if include_forecasts:
            num_models = len(self.performance["performances"])
            with self.source_job.artifact(cache=False) as artifact:
                (target / "forecasts").mkdir()
                for i in range(num_models):
                    (target / "forecasts" / f"model_{i:02}").mkdir()
                    shutil.copyfile(
                        artifact.path
                        / "predictions"
                        / f"model_{i}"
                        / "values.npy",
                        target / "forecasts" / f"model_{i:02}" / "values.npy",
                    )
                    shutil.copyfile(
                        artifact.path
                        / "predictions"
                        / f"model_{i}"
                        / "metadata.npz",
                        target
                        / "forecasts"
                        / f"model_{i:02}"
                        / "metadata.npz",
                    )

        # Finally check that saving all data worked as expected
        assert _check_all_data_available(target)


# -------------------------------------------------------------------------------------------------


def load_jobs_from_analysis(analysis: Analysis) -> list[Job]:
    """
    Returns all jobs loaded from the provided analysis object.

    Args:
        cache: W

    Note:
        This function might take a long time if job logs are not cached. They are downloaded
        sequentially.
    """
    return [Job.from_training_job(job) for job in analysis]


def load_jobs_from_directory(directory: Path) -> list[Job]:
    """
    Returns all jobs stored in the provided directory, assuming that the
    directory is structured as `<model>/<dataset>/<job>`.
    """
    return [
        Job.load(directory / model / dataset / job)
        for model in os.listdir(directory)
        if (directory / model).is_dir() and model in MODEL_REGISTRY
        for dataset in os.listdir(directory / model)
        if (directory / model / dataset).is_dir()
        for job in os.listdir(directory / model / dataset)
        if (directory / model / dataset / job).is_dir()
    ]


# -------------------------------------------------------------------------------------------------


def _extract_configuration(job: TrainingJob) -> dict[str, Any]:
    model = job.hyperparameters["model"]

    hyperparameters = {}
    if "context_length_multiple" in job.hyperparameters:
        hyperparameters["context_length_multiple"] = job.hyperparameters[
            "context_length_multiple"
        ]
    for key, value in job.hyperparameters.items():
        if key.startswith(f"{model}_"):
            hyperparameters[key[len(model) + 1 :]] = value

    return {
        "seed": job.hyperparameters["seed"],
        "hyperparameters": hyperparameters,
    }


def _extract_performance(job: TrainingJob) -> dict[str, Any]:
    num_models = len(job.metrics["training_time"])

    # We need to keep the "mean_weighted_quantile_loss" for legacy experiments
    hierarchy = {
        "training": ["training_time", "num_gradient_updates"],
        "evaluation": [
            "train_loss",
            "val_loss",
            "val_mean_weighted_quantile_loss",
            "val_ncrps",
        ],
        "testing": [
            "mase",
            "smape",
            "nrmse",
            "nd",
            "mean_weighted_quantile_loss",
            "ncrps",
        ],
    }
    rename = {
        "training_time": "duration",
        "mean_weighted_quantile_loss": "ncrps",
        "val_mean_weighted_quantile_loss": "val_ncrps",
    }
    integer_metrics = {"num_gradient_updates"}

    performances = [
        {
            group: {
                rename.get(item, item): (
                    int(job.metrics[item][i].item())
                    if item in integer_metrics
                    else job.metrics[item][i].item()
                )
                for item in items
                if item in job.metrics
            }
            for group, items in hierarchy.items()
            if any(len(job.metrics[item]) > 0 for item in items)
        }
        for i in range(num_models)
    ]

    result = {
        "meta": {
            "num_model_parameters": int(
                job.metrics["num_model_parameters"][0].item()
            ),
            "latency": np.mean(job.metrics["latency"]).item(),
        },
        "performances": performances,
    }

    assert len(result["performances"]) in (1, 11)
    assert all(
        all(not math.isnan(v) for v in vv.values())
        for p in result["performances"]
        for vv in p.values()  # type: ignore
    )

    return result


def _check_all_data_available(target: Path) -> bool:
    return (
        (target / "config.json").exists()
        and (target / "performance.json").exists()
        and (target / "forecasts").exists()
        and len(os.listdir(target / "forecasts")) in (1, 11)
    )
