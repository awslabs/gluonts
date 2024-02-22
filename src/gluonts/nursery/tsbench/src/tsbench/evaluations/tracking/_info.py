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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from tsbench.config import Config, MODEL_REGISTRY, ModelConfig, TrainConfig
from tsbench.config.dataset import get_dataset_config
from tsbench.config.model import get_model_config
from tsbench.constants import DEFAULT_DATA_PATH
from tsbench.evaluations.metrics import Metric, Performance
from .job import Job

ValidationMetric = Literal["train_loss", "val_loss", "val_ncrps"]


@dataclass
class ValidationScores:
    """
    Scores obtained during validation.
    """

    ncrps: float
    loss: float


@dataclass
class JobInfo:
    """
    The job info class aggregates all information available for a particular
    model configuration.

    It also provides the underlying training jobs. Lastly, it provides the
    indices of the models chosen from the training job to extract forecasts.
    """

    config: Config[ModelConfig]
    performance: Performance
    val_scores: Optional[ValidationScores]
    jobs: List[Job]
    model_indices: List[int]


# -------------------------------------------------------------------------------------------------


def extract_job_infos(
    training_jobs: List[Job],
    validation_metric: Optional[ValidationMetric],
    group_seeds: bool,
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
) -> List[JobInfo]:
    """
    Returns a list of the job information objects available for all training
    jobs provided.
    """
    # We group the jobs by hyperparameters, excluding the seed
    if group_seeds:
        grouped_jobs = defaultdict(list)
        for job in training_jobs:
            hypers = {
                "model": job.model,
                "dataset": job.dataset,
                **job.hyperparameters,
            }
            grouped_jobs[tuple(sorted(hypers.items()))].append(job)
        all_jobs = grouped_jobs.values()
    else:
        all_jobs = [[job] for job in training_jobs]

    # Then, we can instantiate the info objects by iterating over groups of jobs
    runs = []
    for jobs in tqdm(all_jobs):
        ref_job = jobs[0]
        model_name = ref_job.model
        base_hyperparams = {**ref_job.hyperparameters}

        # First, we reconstruct the training times
        if issubclass(MODEL_REGISTRY[model_name], TrainConfig):
            training_fractions = [1 / 81, 1 / 27] + [
                i / 9 for i in range(1, 10)
            ]
        else:
            training_fractions = [0]

        assert all(
            len(job.metrics) == len(training_fractions) for job in jobs
        ), "Job does not provide sufficiently many models."

        # Then, we iterate over the Hyperband training times
        if len(training_fractions) == 1:
            training_fraction_indices = [0]
        else:
            training_fraction_indices = [0, 1, 2, 4, 10]

        # Then, we iterate over all training times, construct the hyperparameters and collect
        # the performane metrics
        for i in training_fraction_indices:
            # Create the config object
            hyperparams = {
                **base_hyperparams,
                "training_fraction": training_fractions[i],
            }
            model_config = get_model_config(model_name, **hyperparams)
            config = Config(
                model_config, get_dataset_config(ref_job.dataset, data_path)
            )

            # Get the indices of the models that should be used to derive the performance
            if validation_metric is None or len(training_fractions) == 1:
                # If the model does not require training, or we don't look at the validation
                # performance, we just choose the current index
                choices = [i] * len(jobs)
            else:
                # Otherwise, we get the minimum value for the metric up to this point in time
                choices = [
                    np.argmin(
                        [
                            p["evaluation"][validation_metric]
                            for p in job.metrics
                        ][: i + 1]
                    ).item()
                    for job in jobs
                ]

            # Get the performances of the chosen models
            performances = [
                job.performances[choice] for choice, job in zip(choices, jobs)
            ]

            # And average the performance
            averaged_performance = Performance(
                **{
                    metric: Metric(
                        np.mean(
                            [getattr(p, metric).mean for p in performances]
                        ),
                        np.std(
                            [getattr(p, metric).mean for p in performances]
                        ),
                    )
                    for metric in Performance.metrics()
                }
            )

            # Get validation scores if available
            try:
                val_ncrps = np.mean(
                    [
                        job.metrics[c]["evaluation"]["val_ncrps"]
                        for (job, c) in zip(jobs, choices)
                    ]
                )
                val_loss = np.mean(
                    [
                        job.metrics[c]["evaluation"]["val_loss"]
                        for (job, c) in zip(jobs, choices)
                    ]
                ).item()
                val_scores = ValidationScores(val_ncrps, val_loss)
            except KeyError:
                val_scores = None

            # Initialize the info object
            runs.append(
                JobInfo(
                    config, averaged_performance, val_scores, jobs, choices
                )
            )

    return runs
