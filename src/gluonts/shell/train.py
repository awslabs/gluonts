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
import multiprocessing
from typing import Any, Optional, Type, Union

# Third-party imports
import numpy as np

# First-party imports
import gluonts
from gluonts.core import fqname_for
from gluonts.core.component import check_gpu_support
from gluonts.core.serde import dump_code
from gluonts.dataset.common import Dataset
from gluonts.evaluation import Evaluator, backtest
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.support.util import maybe_len
from gluonts.transform import FilterTransformation, TransformedDataset

# Relative imports
from .sagemaker import TrainEnv

logger = logging.getLogger(__name__)


def log_metric(metric: str, value: Any) -> None:
    """
    Emits a log message with a ``value`` for a specific ``metric``.

    Parameters
    ----------
    metric
        The name of the metric to be reported.
    value
        The metric value to be reported.
    """
    logger.info(f"gluonts[{metric}]: {dump_code(value)}")


def run_train_and_test(
    env: TrainEnv, forecaster_type: Type[Union[Estimator, Predictor]]
) -> None:
    check_gpu_support()

    # train_stats = calculate_dataset_statistics(env.datasets["train"])
    # log_metric("train_dataset_stats", train_stats)

    forecaster_fq_name = fqname_for(forecaster_type)
    forecaster_version = forecaster_type.__version__

    logger.info(f"Using gluonts v{gluonts.__version__}")
    logger.info(f"Using forecaster {forecaster_fq_name} v{forecaster_version}")

    forecaster = forecaster_type.from_inputs(
        env.datasets["train"], **env.hyperparameters
    )

    logger.info(
        f"The forecaster can be reconstructed with the following expression: "
        f"{dump_code(forecaster)}"
    )

    logger.info(
        "Using the following data channels: "
        f"{', '.join(name for name in ['train', 'validation', 'test'] if name in env.datasets)}"
    )

    if isinstance(forecaster, Predictor):
        predictor = forecaster
    else:
        predictor = run_train(
            forecaster, env.datasets["train"], env.datasets.get("validation"), env.train_auxillary_parameters
        )


    predictor.serialize(env.path.model)

    if "test" in env.datasets:
        run_test(env, predictor, env.datasets["test"])


def run_train(
    forecaster: Estimator,
    train_dataset: Dataset,
    validation_dataset: Optional[Dataset],
    train_auxillary_parameters: dict,
) -> Predictor:
    num_workers = (
        int(train_auxillary_parameters["num_workers"]) if "num_workers" in train_auxillary_parameters
        else min(4, int(np.ceil(np.sqrt(multiprocessing.cpu_count()))))
    )
    num_batches_shuffle = (
        int(train_auxillary_parameters["num_batches_shuffle"]) if "num_batches_shuffle" in train_auxillary_parameters
        else None
    )

    return forecaster.train(training_data=train_dataset,
                            validation_data=validation_dataset,
                            num_workers=num_workers,
                            num_batches_shuffle=num_batches_shuffle)


def run_test(
    env: TrainEnv, predictor: Predictor, test_dataset: Dataset
) -> None:
    len_original = maybe_len(test_dataset)

    test_dataset = TransformedDataset(
        base_dataset=test_dataset,
        transformations=[
            FilterTransformation(
                lambda x: x["target"].shape[-1] > predictor.prediction_length
            )
        ],
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

    agg_metrics, _item_metrics = Evaluator()(
        ts_iterator=ts_it,
        fcst_iterator=forecast_it,
        num_series=len(test_dataset),
    )

    # we only log aggregate metrics for now as item metrics may be very large
    for name, score in agg_metrics.items():
        logger.info(f"#test_score ({env.current_host}, {name}): {score}")
