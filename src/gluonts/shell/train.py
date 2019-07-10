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
from gluonts.core import log
from gluonts.core.component import check_gpu_support
from gluonts.evaluation import Evaluator, backtest
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.transform import Dataset, FilterTransformation, TransformedDataset

# Relative imports
from .sagemaker import TrainEnv


logger = logging.getLogger("gluonts.train")


def log_metrics(env, metrics):
    for name, score in metrics.items():
        print(f"#test_score ({env.current_host}, {name}): {score}")


def run_train_and_test(
    env: TrainEnv, forecaster_type: Type[Union[Estimator, Predictor]]
) -> None:
    check_gpu_support()

    forecaster = forecaster_type.from_hyperparameters(**env.hyperparameters)

    if isinstance(forecaster, Predictor):
        predictor = forecaster
    else:
        predictor = run_train(forecaster, env.datasets["train"])

    predictor.serialize(env.path.model)

    if "test" in env.datasets:
        test_dataset = prepare_test_dataset(
            env.datasets["test"],
            prediction_length=forecaster.prediction_length,
        )
        run_test(env, predictor, test_dataset)


def run_train(forecaster, train_dataset) -> Predictor:
    assert isinstance(forecaster, Estimator)
    log.metric('train_dataset_stats', train_dataset.calc_stats())

    return forecaster.train(train_dataset)


def run_test(
    env: TrainEnv, predictor: Predictor, test_dataset: Dataset
) -> None:
    forecast_it, ts_it = backtest.make_evaluation_predictions(
        test_dataset, predictor=predictor, num_eval_samples=100
    )
    agg_metrics, _item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(test_dataset)
    )

    # we only log aggregate metrics for now as item metrics may be
    # very large
    log_metrics(env, agg_metrics)


def prepare_test_dataset(dataset: Dataset, prediction_length: int) -> Dataset:
    test_dataset = TransformedDataset(
        dataset,
        transformations=[
            FilterTransformation(
                lambda el: el['target'].shape[-1] > prediction_length
            )
        ],
    )

    len_orig = len(dataset)
    len_filtered = len(test_dataset)
    if len_orig > len_filtered:
        log.logger.warning(
            'Not all time-series in the test-channel have '
            'enough data to be used for evaluation. Proceeding with '
            f'{len_filtered}/{len_orig} '
            f'(~{int(len_filtered / len_orig * 100)}%) items.'
        )
    return test_dataset
