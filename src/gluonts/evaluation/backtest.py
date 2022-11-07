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
import re
from typing import Dict, Iterator, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

import gluonts  # noqa
from gluonts.core.serde import load_code
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.dataset.stat import (
    DatasetStatistics,
    calculate_dataset_statistics,
)
from gluonts.dataset.util import period_index
from gluonts.evaluation import Evaluator
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from gluonts.itertools import maybe_len


def _to_dataframe(input_label: Tuple[DataEntry, DataEntry]) -> pd.DataFrame:
    """
    Turn a pair of consecutive (in time) data entries into a dataframe.
    """
    start = input_label[0][FieldName.START]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)


def make_evaluation_predictions(
    dataset: Dataset,
    predictor: Predictor,
    num_samples: int = 100,
) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
    """
    Returns predictions for the trailing prediction_length observations of the
    given time series, using the given predictor.

    The predictor will take as input the given time series without the trailing
    prediction_length observations.

    Parameters
    ----------
    dataset
        Dataset where the evaluation will happen. Only the portion excluding
        the prediction_length portion is used when making prediction.
    predictor
        Model used to draw predictions.
    num_samples
        Number of samples to draw on the model when evaluating. Only
        sampling-based models will use this.

    Returns
    -------
    Tuple[Iterator[Forecast], Iterator[pd.Series]]
        A pair of iterators, the first one yielding the forecasts, and the
        second one yielding the corresponding ground truth series.
    """

    window_length = predictor.prediction_length + predictor.lead_time
    _, test_template = split(dataset, offset=-window_length)
    test_data = test_template.generate_instances(window_length)

    return (
        predictor.predict(test_data.input, num_samples=num_samples),
        map(_to_dataframe, test_data),
    )


train_dataset_stats_key = "train_dataset_stats"
test_dataset_stats_key = "test_dataset_stats"
estimator_key = "estimator"
agg_metrics_key = "agg_metrics"


def serialize_message(logger, message: str, variable):
    logger.info(f"gluonts[{message}]: {variable}")


def backtest_metrics(
    test_dataset: Dataset,
    predictor: Predictor,
    evaluator=Evaluator(
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ),
    num_samples: int = 100,
    logging_file: Optional[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Parameters
    ----------
    test_dataset
        Dataset to use for testing.
    predictor
        The predictor to test.
    evaluator
        Evaluator to use.
    num_samples
        Number of samples to use when generating sample-based forecasts. Only
        sampling-based models will use this.
    logging_file
        If specified, information of the backtest is redirected to this file.

    Returns
    -------
    Tuple[dict, pd.DataFrame]
        A tuple of aggregate metrics and metrics per time series obtained by
        training `forecaster` on `train_dataset` and evaluating the resulting
        `evaluator` provided on the `test_dataset`.
    """

    if logging_file is not None:
        log_formatter = logging.Formatter(
            "[%(asctime)s %(levelname)s %(thread)d] %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(logging_file)
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)
    else:
        logger = logging.getLogger(__name__)

    test_statistics = calculate_dataset_statistics(test_dataset)
    serialize_message(logger, test_dataset_stats_key, test_statistics)

    forecast_it, ts_it = make_evaluation_predictions(
        test_dataset, predictor=predictor, num_samples=num_samples
    )

    agg_metrics, item_metrics = evaluator(
        ts_it, forecast_it, num_series=maybe_len(test_dataset)
    )

    # we only log aggregate metrics for now as item metrics may be very large
    for name, value in agg_metrics.items():
        serialize_message(logger, f"metric-{name}", value)

    if logging_file is not None:
        # Close the file handler to avoid letting the file open.
        # https://stackoverflow.com/questions/24816456/python-logging-wont-shutdown
        logger.removeHandler(handler)
        del logger, handler

    return agg_metrics, item_metrics


# TODO does it make sense to have this then?
class BacktestInformation(NamedTuple):
    train_dataset_stats: DatasetStatistics
    test_dataset_stats: DatasetStatistics
    estimator: Estimator
    agg_metrics: Dict[str, float]

    @staticmethod
    def make_from_log(log_file):
        with open(log_file) as f:
            return BacktestInformation.make_from_log_contents(
                "\n".join(f.readlines())
            )

    @staticmethod
    def make_from_log_contents(log_contents):
        messages = dict(re.findall(r"gluonts\[(.*)\]: (.*)", log_contents))

        # avoid to fail if a key is missing for instance in the case a run did
        # not finish so that we can still get partial information
        try:
            return BacktestInformation(
                train_dataset_stats=eval(
                    messages[train_dataset_stats_key]
                ),  # TODO: use load
                test_dataset_stats=eval(
                    messages[test_dataset_stats_key]
                ),  # TODO: use load
                estimator=load_code(messages[estimator_key]),
                agg_metrics={
                    k: load_code(v)
                    for k, v in messages.items()
                    if k.startswith("metric-") and v != "nan"
                },
            )
        except Exception as error:
            logging.error(error)
            return None
