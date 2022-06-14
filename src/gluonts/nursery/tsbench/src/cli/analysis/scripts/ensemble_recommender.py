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

# pylint: disable=missing-function-docstring
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from sacred import Experiment
from tsbench.analysis.recommender import EnsembleRecommenderAnalyzer
from tsbench.constants import DEFAULT_ENSEMBLE_EVALUATIONS_PATH
from tsbench.evaluations.tracking import EnsembleTracker
from tsbench.recommender import create_ensemble_recommender
from tsbench.surrogate import create_ensemble_surrogate

ex = Experiment()


@ex.config
def experiment_config():
    # pylint: disable=unused-variable
    experiment = "test"  # type: ignore
    evaluations_path = str(DEFAULT_ENSEMBLE_EVALUATIONS_PATH)  # type: ignore

    recommender = "pareto"  # type: ignore
    num_recommendations = 20  # type: ignore
    objectives = "ncrps_mean,latency_mean"  # type: ignore
    focus_objective = None  # type: ignore

    surrogate = {  # type: ignore
        "name": "deepset",
        "outputs": {
            "imputation": False,
        },
        "deepset": {
            "objective": "ranking",
            "discount": "linear",
            "hidden_layer_sizes": [32, 32],
            "weight_decay": 0.01,
            "dropout": 0.0,
        },
    }


@ex.automain
def main(
    _seed: int,
    evaluations_path: str,
    recommender: str,
    num_recommendations: int,
    objectives: str,
    focus_objective: Optional[str],
    surrogate: Dict[str, Any],
):
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # First, get the tracker
    print("Fetching the data...")
    tracker = EnsembleTracker(Path(evaluations_path))

    # Then, potentially initialize the surrogate
    recommender_args: Dict[str, Any] = {
        "objectives": objectives.split(","),
        "focus": focus_objective,
        "recommends_ensembles": True,
    }

    print("Initializing the surrogate...")
    surrogate_metrics = [
        m
        for m in objectives.split(",")
        if (
            not m.startswith("latency")
            and not m.startswith("num_model_parameters")
        )
        or not surrogate["outputs"]["imputation"]
    ]
    recommender_args["surrogate"] = create_ensemble_surrogate(
        surrogate["name"],
        predict=surrogate_metrics,
        tracker=tracker,
        input_flags={},
        impute_simulatable=surrogate["outputs"]["imputation"],
        **(
            surrogate[surrogate["name"]]
            if surrogate["name"] in surrogate
            else {}
        )
    )

    # Then, we can create the recommender
    print("Initializing the recommender...")
    recommender_instance = create_ensemble_recommender(
        recommender, **recommender_args
    )

    # And evaluate it
    print("Evaluating the recommender...")
    evaluator = EnsembleRecommenderAnalyzer(
        tracker,
        recommender_instance,
        num_recommendations=num_recommendations,
    )
    recommendations = evaluator.run()

    # Eventually, we store the results
    print("Storing the results...")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "recommendations.pickle"
        with path.open("wb+") as f:
            pickle.dump(recommendations, f)
        ex.add_artifact(path, content_type="application/octet-stream")
