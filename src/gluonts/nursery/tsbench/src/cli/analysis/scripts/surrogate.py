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
import tempfile
from pathlib import Path
from typing import Any, Dict
import pytorch_lightning as pl
from sacred import Experiment
from tsbench.analysis import SurrogateAnalyzer
from tsbench.constants import DEFAULT_DATA_PATH, DEFAULT_EVALUATIONS_PATH
from tsbench.evaluations.tracking import ModelTracker
from tsbench.surrogate import create_surrogate

ex = Experiment()


@ex.config
def experiment_config():
    # pylint: disable=unused-variable
    experiment = "test"  # type: ignore
    data_path = str(DEFAULT_DATA_PATH)  # type: ignore
    evaluations_path = str(DEFAULT_EVALUATIONS_PATH)  # type: ignore
    surrogate = "nonparametric"  # type: ignore

    metrics = "ncrps_mean,latency_mean"  # type: ignore
    inputs = {  # type: ignore
        "use_simple_dataset_features": False,
        "use_seasonal_naive_performance": False,
        "use_catch22_features": False,
    }
    outputs = {  # type: ignore
        "normalization": "quantile",
        "imputation": False,
    }

    xgboost = {  # type: ignore
        "objective": "regression",
    }
    autogluon = {  # type: ignore
        "time_limit": 10,
    }
    mlp = {  # type: ignore
        "objective": "ranking",
        "discount": "linear",
        "hidden_layer_sizes": [32, 32],
        "weight_decay": 0.01,
        "dropout": 0.0,
    }


@ex.automain
def main(
    _config: Dict[str, Any],
    _seed: int,
    data_path: str,
    evaluations_path: str,
    surrogate: str,
    metrics: str,
    inputs: Dict[str, bool],
    outputs: Dict[str, Any],
):
    print(_config)

    pl.seed_everything(_seed)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # First, get the tracker
    print("Fetching the data...")
    tracker = ModelTracker.from_directory(
        Path(evaluations_path), data_path=Path(data_path)
    )

    # Then, initialize the surrogate
    print("Initializing the surrogate...")
    metrics_list = metrics.split(",")
    surrogate_instance = create_surrogate(
        surrogate,
        tracker=tracker,
        predict=metrics_list,
        input_flags=inputs,
        output_normalization=outputs["normalization"],
        impute_simulatable=outputs["imputation"],
        **(_config[surrogate] if surrogate in _config else {})
    )

    # And evaluate it
    print("Evaluating the surrogate...")
    evaluator = SurrogateAnalyzer(
        surrogate_instance, tracker=tracker, metrics=metrics_list
    )
    result = evaluator.run()

    print(result.mean())

    # Eventually, we store the results
    print("Storing the results...")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "results.parquet"
        result.to_parquet(str(path))
        ex.add_artifact(path, content_type="application/octet-stream")
