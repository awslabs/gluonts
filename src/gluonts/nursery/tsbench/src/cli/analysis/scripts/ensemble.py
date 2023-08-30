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
from typing import Optional
import numpy as np
import torch
from sacred import Experiment
from tsbench.analysis import EnsembleAnalyzer
from tsbench.config import MODEL_REGISTRY
from tsbench.constants import DEFAULT_DATA_PATH, DEFAULT_EVALUATIONS_PATH
from tsbench.evaluations.tracking import ModelTracker
from tsbench.forecasts import EnsembleWeighting

ex = Experiment()


@ex.config
def experiment_config():
    # pylint: disable=unused-variable
    experiment = "test"  # type: ignore
    data_path = str(DEFAULT_DATA_PATH)  # type: ignore
    evaluations_path = str(DEFAULT_EVALUATIONS_PATH)  # type: ignore

    weighting = "uniform"  # type: ignore
    size = 10  # type: ignore
    model_class = None  # type: ignore


@ex.automain
def main(
    _seed: int,
    data_path: str,
    evaluations_path: str,
    weighting: EnsembleWeighting,
    size: int,
    model_class: Optional[str],
):
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # First, get the tracker
    print("Fetching the data...")
    tracker = ModelTracker.from_directory(
        Path(evaluations_path), data_path=Path(data_path)
    )

    # Evaluate the ensemble that can be built
    print("Evaluating the ensemble...")
    evaluator = EnsembleAnalyzer(
        tracker,
        ensemble_size=size,
        ensemble_weighting=weighting,
        config_class=MODEL_REGISTRY[model_class]
        if model_class is not None
        else None,
    )
    df, configs = evaluator.run()

    # Eventually, we store the results
    print("Storing the results...")
    with tempfile.TemporaryDirectory() as d:
        df_path = Path(d) / "results.parquet"
        df.to_parquet(str(df_path))
        ex.add_artifact(df_path, content_type="application/octet-stream")

        config_path = Path(d) / "configs.pickle"
        with config_path.open("wb+") as f:
            pickle.dump(configs, f)
        ex.add_artifact(config_path, content_type="application/octet-stream")
