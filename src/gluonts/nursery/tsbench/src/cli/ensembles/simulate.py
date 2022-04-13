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

import pickle
import random
from functools import partial
from itertools import combinations, product
from pathlib import Path
from typing import cast, List, Tuple
import click
from tsbench.analysis import EnsembleAnalyzer
from tsbench.analysis.utils import num_fitting_processes, run_parallel
from tsbench.config import DatasetConfig, ModelConfig, TrainConfig
from tsbench.constants import DEFAULT_EVALUATIONS_PATH
from tsbench.evaluations.metrics import Performance
from tsbench.evaluations.tracking import ModelTracker
from ._main import ensembles


@ensembles.command(short_help="Simulate the performance of ensembles.")
@click.option(
    "--output_path",
    type=click.Path(),
    default=Path.home() / "ensembles" / "evaluation.pickle",
    show_default=True,
    help="The local file where evaluated ensembles should be written to.",
)
@click.option(
    "--data_path",
    type=click.Path(exists=True),
    default=Path.home() / "data" / "datasets",
    show_default=True,
    help="The local path where the datasets are stored.",
)
@click.option(
    "--evaluations_path",
    type=click.Path(exists=True),
    default=DEFAULT_EVALUATIONS_PATH,
    show_default="True",
    help="The local path where offline evaluations can be found.",
)
@click.option(
    "--max_ensemble_size",
    default=10,
    help=(
        "The maximum number of ensemble members for any ensemble that is"
        " evaluated."
    ),
)
@click.option(
    "--default_samples",
    default=0,
    show_default=True,
    help=(
        "The number of ensembles to sample which combine only default"
        " configurations of all models. If set to -1, uses all available"
        " configurations."
    ),
)
@click.option(
    "--hyperensemble_samples",
    default=0,
    show_default=True,
    help=(
        "The number of ensembles to sample which constitute hyperensembles. If"
        " set to -1, usesall hyperensembles that can be built."
    ),
)
@click.option(
    "--random_samples",
    default=0,
    show_default=True,
    help=(
        "The number of ensembles to sample which combine random configurations"
        " found in theoffline evaluations."
    ),
)
@click.option(
    "--sample_datasets",
    default=True,
    show_default=True,
    help=(
        "Whether to sample a single random dataset for each ensemble being"
        " evaluated or toevaluate each ensemble on all available datasets."
    ),
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    help="Seed for reproducible sampling.",
)
def simulate(
    data_path: str,
    evaluations_path: str,
    output_path: str,
    max_ensemble_size: int,
    default_samples: int,
    hyperensemble_samples: int,
    random_samples: int,
    sample_datasets: bool,
    seed: int,
):
    """
    Simulates the performance of various ensembles.

    The ensembles are built from configurations (i.e. model types and
    hyperparameters) for which offline evaluations are available.
    """
    assert any(
        [default_samples != 0, hyperensemble_samples != 0, random_samples != 0]
    ), "No samples are specified."

    random.seed(seed)

    # Load the experiments
    print("Loading experiments...")
    tracker = ModelTracker.from_directory(
        Path(evaluations_path), data_path=Path(data_path)
    )

    # Sample configurations
    print("Sampling configurations...")
    unique_configurations = tracker.unique_model_configs()
    default_configurations = [
        cast(ModelConfig, c)
        for c in unique_configurations
        if not isinstance(c, TrainConfig) or c == c.__class__()
    ]
    choices: List[Tuple[ModelConfig]] = []

    # If desired, we combine all base configurations into ensembles of sizes between 2 and the
    # provided maximum. Then, we potentially sample from this collection. For 13 default
    # configurations and a maximum ensemble size of 10, this results in 8,086 ensembles.
    if default_samples != 0:
        available_ensembles = [
            combination
            for i in range(2, max_ensemble_size + 1)
            for combination in combinations(default_configurations, i)
        ]
        if default_samples == -1:
            choices.extend(available_ensembles)
        else:
            choices.extend(random.sample(available_ensembles, default_samples))

    if hyperensemble_samples != 0:
        available_ensembles = []
        for config in default_configurations:
            all_configs = [
                c
                for c in unique_configurations
                if isinstance(c, config.__class__)
                and (
                    not isinstance(c, TrainConfig) or c.training_fraction == 1
                )
            ]
            if len(all_configs) == 1:
                continue
            hyper_ensembles = [
                combination
                for i in range(2, max_ensemble_size + 1)
                for combination in combinations(all_configs, i)
            ]
            available_ensembles.extend(hyper_ensembles)

        if hyperensemble_samples == -1:
            choices.extend(available_ensembles)
        else:
            choices.extend(
                random.sample(available_ensembles, hyperensemble_samples)
            )

    # Then, we add some randomly sampled ensembles of model configurations
    for _ in range(random_samples):
        ensemble_size = random.randrange(2, max_ensemble_size + 1)
        configs = random.sample(unique_configurations, ensemble_size)
        choices.append(tuple(configs))

    # Then, we either evaluate each chosen configuration on all datasets or on a randomly sampled
    # one.
    datasets = list(
        {c.dataset for c in tracker.get_evaluations().configurations}
    )
    if sample_datasets:
        evaluations = [
            (model_config, random.choice(datasets)) for model_config in choices
        ]
    else:
        evaluations = list(product(choices, datasets))

    # Eventually, we can evaluate the ensembles that we have sampled
    print("Evaluating ensembles...")
    evaluator = EnsembleAnalyzer(tracker)
    results = run_parallel(
        partial(_evaluate_ensemble, evaluator=evaluator),
        evaluations,
        num_fitting_processes(cpus_per_process=1, memory_per_process=8),
    )

    # Afterwards, we can store all configurations along with their results. For now, we are just
    # storing them as pickled objects.
    with Path(output_path).open("wb+") as f:
        pickle.dump(
            [
                {
                    "configurations": list(evaluation[0]),
                    "dataset": evaluation[1],
                    "performance": result,
                }
                for evaluation, result in zip(evaluations, results)
            ],
            f,
        )


def _evaluate_ensemble(
    ensemble: Tuple[Tuple[ModelConfig], DatasetConfig],
    evaluator: EnsembleAnalyzer,
) -> Performance:
    return evaluator.get_ensemble_performance(
        list(ensemble[0]), ensemble[1], num_samples=3
    )
