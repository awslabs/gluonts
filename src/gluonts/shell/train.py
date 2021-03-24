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

import json
import logging
import multiprocessing
from typing import Any, Optional, Type, Union

import gluonts
from gluonts.core import fqname_for
from gluonts.core.serde import dump_code
from gluonts.dataset.common import Dataset
from gluonts.evaluation import Evaluator, backtest
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import Quantile
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.model.predictor import Predictor
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.support.util import maybe_len
from gluonts.transform import FilterTransformation, TransformedDataset

from .env import TrainEnv

multiprocessing.set_start_method("spawn", force=True)
logger = logging.getLogger(__name__)


def log_metric(metric: str, value: Any) -> None:
    logger.info(f"gluonts[{metric}]: {dump_code(value)}")


def log_version(forecaster_type):
    name = fqname_for(forecaster_type)
    version = forecaster_type.__version__

    logger.info(f"Using gluonts v{gluonts.__version__}")
    logger.info(f"Using forecaster {name} v{version}")


def run_train_and_test(
    env: TrainEnv, forecaster_type: Type[Union[Estimator, Predictor]]
) -> None:
    log_version(forecaster_type)

    logger.info(
        "Using the following data channels: %s", ", ".join(env.datasets)
    )

    forecaster = forecaster_type.from_inputs(
        env.datasets["train"], **env.hyperparameters
    )
    logger.info(
        f"The forecaster can be reconstructed with the following expression: "
        f"{dump_code(forecaster)}"
    )

    if isinstance(forecaster, Predictor):
        predictor = forecaster
    else:
        predictor = run_train(
            forecaster=forecaster,
            train_dataset=env.datasets["train"],
            validation_dataset=env.datasets.get("validation"),
            hyperparameters=env.hyperparameters,
        )

    predictor.serialize(env.path.model)

    if "test" in env.datasets:
        run_test(env, predictor, env.datasets["test"], env.hyperparameters)


def run_train(
    forecaster: Estimator,
    train_dataset: Dataset,
    hyperparameters: dict,
    validation_dataset: Optional[Dataset],
) -> Predictor:
    num_workers = (
        int(hyperparameters["num_workers"])
        if "num_workers" in hyperparameters.keys()
        else None
    )
    shuffle_buffer_length = (
        int(hyperparameters["shuffle_buffer_length"])
        if "shuffle_buffer_length" in hyperparameters.keys()
        else None
    )
    num_prefetch = (
        int(hyperparameters["num_prefetch"])
        if "num_prefetch" in hyperparameters.keys()
        else None
    )
    if isinstance(forecaster, GluonEstimator):
        return forecaster.train(
            training_data=train_dataset,
            validation_data=validation_dataset,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            shuffle_buffer_length=shuffle_buffer_length,
        )
    else:
        return forecaster.train(
            training_data=train_dataset, validation_data=validation_dataset
        )


def run_test(
    env: TrainEnv,
    predictor: Predictor,
    test_dataset: Dataset,
    hyperparameters: dict,
) -> None:
    len_original = maybe_len(test_dataset)

    test_dataset = TransformedDataset(
        test_dataset,
        FilterTransformation(
            lambda x: x["target"].shape[-1] > predictor.prediction_length
        ),
    )

    len_filtered = len(test_dataset)

    if len_original is not None and len_original > len_filtered:
        logger.warning(
            f"Not all time-series in the test-channel have "
            f"enough data to be used for evaluation. Proceeding with "
            f"{len_filtered}/{len_original} "
            f"(~{int(len_filtered / len_original * 100)}%) items."
        )

    forecast_it, ts_it = backtest.make_evaluation_predictions(
        dataset=test_dataset, predictor=predictor, num_samples=100
    )

    test_quantiles = (
        [
            Quantile.parse(quantile).name
            for quantile in hyperparameters["test_quantiles"]
        ]
        if "test_quantiles" in hyperparameters.keys()
        else None
    )

    if isinstance(predictor, RepresentableBlockPredictor) and isinstance(
        predictor.forecast_generator, QuantileForecastGenerator
    ):
        predictor_quantiles = predictor.forecast_generator.quantiles
        if test_quantiles is None:
            test_quantiles = predictor_quantiles
        elif not set(test_quantiles).issubset(set(predictor_quantiles)):
            logger.warning(
                f"Some of the evaluation quantiles `{test_quantiles}` are "
                f"not in the computed quantile forecasts `{predictor_quantiles}`."
            )
            test_quantiles = predictor_quantiles

    if test_quantiles is not None:
        logger.info(f"Using quantiles `{test_quantiles}` for evaluation.")
        evaluator = Evaluator(quantiles=test_quantiles)
    else:
        evaluator = Evaluator()

    agg_metrics, item_metrics = evaluator(
        ts_iterator=ts_it,
        fcst_iterator=forecast_it,
        num_series=len(test_dataset),
    )

    # we only log aggregate metrics for now as item metrics may be very large
    for name, score in agg_metrics.items():
        logger.info(f"#test_score ({env.current_host}, {name}): {score}")

    # store metrics
    with open(env.path.model / "agg_metrics.json", "w") as agg_metric_file:
        json.dump(agg_metrics, agg_metric_file)
    with open(env.path.model / "item_metrics.csv", "w") as item_metrics_file:
        item_metrics.to_csv(item_metrics_file, index=False)
