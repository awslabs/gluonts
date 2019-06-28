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

import os
from pydoc import locate

# First-party imports
from gluonts.core import log
from gluonts.core.component import check_gpu_support, from_hyperparameters
from gluonts.core.exception import GluonTSFatalError
from gluonts.evaluation import Evaluator, backtest
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.shell.env import SageMakerEnv
from gluonts.transform import FilterTransformation, TransformedDataset


def run(env, forecaster):
    check_gpu_support()

    if isinstance(forecaster, Predictor):
        predictor = forecaster
    else:
        predictor = run_train(env, forecaster, env.datasets["train"])

    if "test" in env.datasets:
        test_dataset = prepare_test_dataset(
            env.datasets["test"],
            prediction_length=forecaster.prediction_length,
        )
        run_test(predictor, test_dataset)


def run_train(env, forecaster, dataset) -> Predictor:
    assert isinstance(forecaster, Estimator)
    log.metric('train_dataset_stats', dataset.calc_stats())

    predictor = forecaster.train(dataset)
    predictor.serialize(env.paths.model)

    return predictor


def run_test(forecaster, test_dataset):
    agg_metrics, _item_metrics = backtest.backtest_metrics(
        train_dataset=None,
        test_dataset=test_dataset,
        forecaster=forecaster,
        evaluator=Evaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
        num_eval_samples=100,
    )

    # we only log aggregate metrics for now as item metrics may be
    # very large
    log.metric("agg_metrics", agg_metrics)


def prepare_test_dataset(dataset, prediction_length):
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
            f'(~{int(len_filtered/len_orig*100)}%) items.'
        )
    return test_dataset


def get_forecaster(forecaster_class=None):
    if 'GLUONTS_ESTIMATOR_CLASS' in os.environ:
        log.logger.info(
            'Picking up GluonTS estimator classname from a '
            '"GLUONTS_ESTIMATOR_CLASS" environment variable.'
        )
        forecaster_class = os.environ['GLUONTS_ESTIMATOR_CLASS']

    elif forecaster_class is not None:
        log.logger.info(
            'Picking up GluonTS estimator classname from an "forecaster_class" '
            'hyperparameter value.'
        )
    else:
        raise GluonTSFatalError(
            'Cannot determine the GluonTS estimator classname (missing variable '
            '"GLUONTS_ESTIMATOR_CLASS").'
        )

    Forecaster = locate(forecaster_class)
    if Forecaster is None:
        raise GluonTSFatalError(
            f'Cannot locate estimator with classname "{forecaster_class}".'
        )
    return Forecaster


def train(path, forecaster_path) -> None:
    env = SageMakerEnv(path)

    # forecaster_path = env.hyperparameters.get("forecaster_class")

    Forecaster = get_forecaster(forecaster_path)
    forecaster = from_hyperparameters(Forecaster, **env.hyperparameters)

    run(env, forecaster)
