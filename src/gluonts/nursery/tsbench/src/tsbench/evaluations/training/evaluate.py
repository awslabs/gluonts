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

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from gluonts.env import env
from gluonts.model.predictor import Predictor
from tsbench.config import DatasetConfig, ModelConfig
from tsbench.config.dataset import DatasetSplit
from tsbench.forecasts import evaluate_forecasts, generate_forecasts
from .logging import log_metric


@dataclass
class FitResult:
    """
    A result object obtained when fitting a model.
    """

    config: ModelConfig
    predictors: List[Predictor]
    training_times: List[float]
    num_model_parameters: int

    def evaluate_predictors(
        self,
        dataset_config: DatasetConfig,
        dataset_split: DatasetSplit,
        directory: Path,
        validation: bool = False,
    ) -> None:
        """
        Evaluates the given predictors on the specified dataset, logs the
        resulting metrics and stores the forecasts in the given directory. The
        forecasts of predictor `i` are stored in the subdirectory `model_<i>`.

        Args:
            dataset_config: The configuration of the dataset to make predictions for.
            dataset_split: The split of the dataset for which to make predictions for.
            directory: The directory where to store forecasts.
            validation: Whether to log only the mean weighted quantile loss with a `val_` prefix.
        """
        length = self.config.max_time_series_length(  # pylint: disable=assignment-from-none
            dataset_config
        )
        if length is not None:
            # pylint: disable=invalid-unary-operand-type
            dataset = [
                {**item, "target": item["target"][-length:]}
                for item in dataset_split.gluonts()
            ]
        else:
            dataset = dataset_split.gluonts()

        for i, predictor in enumerate(self.predictors):
            logging.info(
                "Evaluating predictor %d/%d...", i + 1, len(self.predictors)
            )

            # Evaluate
            with _suppress_stdout_stderr():  # need to do this to suppress Prophet outputs
                prediction, latency = generate_forecasts(
                    predictor,
                    dataset,
                    num_samples=self.config.prediction_samples,
                    parallelize=self.config.prefers_parallel_predictions,
                )
            evaluation = evaluate_forecasts(
                prediction, dataset_split.evaluation()
            )

            # Log the summary and store the predictions
            eval_dir = directory / f"model_{i}"
            eval_dir.mkdir(parents=True, exist_ok=True)

            prediction.save(eval_dir)
            if not validation:
                for metric, value in evaluation.summary.items():
                    log_metric(metric, value)
                log_metric("latency", latency)
            else:
                log_metric(
                    "val_ncprs",
                    evaluation.summary["ncrps"],
                )

    def serialize_predictors(self, directory: Path) -> None:
        """
        Serializes all predictos managed by the fit result into the given
        directory. Predictor `i` is saved to subdirectory `model_<i>`.

        Args:
            directory: The directory where the predictors should be serialized to.
        """
        for i, predictor in enumerate(self.predictors):
            logging.info(
                "Serializing predictor %d/%d...", i + 1, len(self.predictors)
            )
            path = directory / f"model_{i}"
            path.mkdir(parents=True, exist_ok=True)
            self.config.save_predictor(predictor, path)


# -------------------------------------------------------------------------------------------------


class _suppress_stdout_stderr:
    def __init__(self):
        if not env.use_tqdm:
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
            self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        if not env.use_tqdm:
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        if not env.use_tqdm:
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            for fd in self.null_fds + self.save_fds:
                os.close(fd)
