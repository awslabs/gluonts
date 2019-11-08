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

# Standard library imports
import logging
from typing import Type, Union

# First-party imports
import gluonts
from gluonts.core import fqname_for, log
from gluonts.core.component import check_gpu_support
from gluonts.core.serde import dump_code
from gluonts.evaluation import Evaluator, backtest
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.transform import Dataset, FilterTransformation, TransformedDataset

# Relative imports
from .sagemaker import TrainEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S %z]",
)

logger = logging.getLogger("gluonts.train")


def run_train_and_test(
    env: TrainEnv, forecaster_type: Type[Union[Estimator, Predictor]]
) -> None:
    check_gpu_support()

    forecaster_fq_name = fqname_for(forecaster_type)
    forecaster_version = forecaster_type.__version__

    logger.info(f"Using gluonts v{gluonts.__version__}")
    logger.info(f"Using forecaster {forecaster_fq_name} v{forecaster_version}")

    forecaster = forecaster_type.from_hyperparameters(**env.hyperparameters)

    logger.info(
        f"The forecaster can be reconstructed with the following expression: "
        f"{dump_code(forecaster)}"
    )

    if isinstance(forecaster, Predictor):
        predictor = forecaster
    else:
        predictor = run_train(forecaster, env.datasets["train"])

    predictor.serialize(env.path.model)

    if "test" in env.datasets:
        run_test(env, predictor, env.datasets["test"])


def run_train(forecaster: Estimator, train_dataset: Dataset) -> Predictor:
    log.metric("train_dataset_stats", train_dataset.calc_stats())

    return forecaster.train(train_dataset)


def run_test(
    env: TrainEnv, predictor: Predictor, test_dataset: Dataset
) -> None:
    len_original = len(test_dataset)

    test_dataset = TransformedDataset(
        base_dataset=test_dataset,
        transformations=[
            FilterTransformation(
                lambda x: x["target"].shape[-1] > predictor.prediction_length
            )
        ],
    )

    len_filtered = len(test_dataset)

    if len_original > len_filtered:
        logger.warning(
            f"Not all time-series in the test-channel have "
            f"enough data to be used for evaluation. Proceeding with "
            f"{len_filtered}/{len_original} "
            f"(~{int(len_filtered / len_original * 100)}%) items."
        )

    forecast_it, ts_it = backtest.make_evaluation_predictions(
        dataset=test_dataset, predictor=predictor, num_samples=100
    )

    agg_metrics, _item_metrics = Evaluator()(
        ts_iterator=ts_it,
        fcst_iterator=forecast_it,
        num_series=len(test_dataset),
    )

    # we only log aggregate metrics for now as item metrics may be very large
    for name, score in agg_metrics.items():
        logger.info(f"#test_score ({env.current_host}, {name}): {score}")
