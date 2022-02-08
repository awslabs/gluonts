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
from typing import Dict, Iterator, NamedTuple, Optional, Tuple, Union

# Third-party imports
import pandas as pd

from pts.dataset import (
    DataEntry,
    Dataset,
    DatasetStatistics,
    calculate_dataset_statistics,
)
from pts.model import Estimator, Predictor, Forecast

# First-party imports
from pts.transform import AdhocTransform, TransformedDataset
from .evaluator import Evaluator


def make_evaluation_predictions(
    dataset: Dataset, predictor: Predictor, num_samples: int
) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
    """
    Return predictions on the last portion of predict_length time units of the
    target. Such portion is cut before making predictions, such a function can
    be used in evaluations where accuracy is evaluated on the last portion of
    the target.

    Parameters
    ----------
    dataset
        Dataset where the evaluation will happen. Only the portion excluding
        the prediction_length portion is used when making prediction.
    predictor
        Model used to draw predictions.
    num_samples
        Number of samples to draw on the model when evaluating.

    Returns
    -------
    """

    prediction_length = predictor.prediction_length
    freq = predictor.freq

    def add_ts_dataframe(
        data_iterator: Iterator[DataEntry],
    ) -> Iterator[DataEntry]:
        for data_entry in data_iterator:
            data = data_entry.copy()
            index = pd.date_range(
                start=data["start"],
                freq=freq,
                periods=data["target"].shape[-1],
            )
            data["ts"] = pd.DataFrame(
                index=index, data=data["target"].transpose()
            )
            yield data

    def ts_iter(dataset: Dataset) -> pd.DataFrame:
        for data_entry in add_ts_dataframe(iter(dataset)):
            yield data_entry["ts"]

    def truncate_target(data):
        data = data.copy()
        target = data["target"]
        assert (
            target.shape[-1] >= prediction_length
        )  # handles multivariate case (target_dim, history_length)
        data["target"] = target[..., :-prediction_length]
        return data

    # TODO filter out time series with target shorter than prediction length
    # TODO or fix the evaluator so it supports missing values instead (all
    # TODO the test set may be gone otherwise with such a filtering)

    dataset_trunc = TransformedDataset(
        dataset, transformations=[AdhocTransform(truncate_target)]
    )

    return (
        predictor.predict(dataset_trunc, num_samples=num_samples),
        ts_iter(dataset),
    )


train_dataset_stats_key = "train_dataset_stats"
test_dataset_stats_key = "test_dataset_stats"
estimator_key = "estimator"
agg_metrics_key = "agg_metrics"


def serialize_message(logger, message: str, variable):
    logger.info(f"pts[{message}]: {variable}")


def backtest_metrics(
    train_dataset: Optional[Dataset],
    test_dataset: Dataset,
    forecaster: Union[Estimator, Predictor],
    evaluator=Evaluator(
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ),
    num_samples: int = 100,
    logging_file: Optional[str] = None,
):
    """
    Parameters
    ----------
    train_dataset
        Dataset to use for training.
    test_dataset
        Dataset to use for testing.
    forecaster
        An estimator or a predictor to use for generating predictions.
    evaluator
        Evaluator to use.
    num_samples
        Number of samples to use when generating sample-based forecasts.
    logging_file
        If specified, information of the backtest is redirected to this file.

    Returns
    -------
    tuple
        A tuple of aggregate metrics and per-time-series metrics obtained by
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

    if train_dataset is not None:
        train_statistics = calculate_dataset_statistics(train_dataset)
        serialize_message(logger, train_dataset_stats_key, train_statistics)

    test_statistics = calculate_dataset_statistics(test_dataset)
    serialize_message(logger, test_dataset_stats_key, test_statistics)

    if isinstance(forecaster, Estimator):
        serialize_message(logger, estimator_key, forecaster)
        assert train_dataset is not None
        predictor = forecaster.train(train_dataset)
    else:
        predictor = forecaster

    forecast_it, ts_it = make_evaluation_predictions(
        test_dataset, predictor=predictor, num_samples=num_samples
    )

    agg_metrics, item_metrics = evaluator(
        ts_it, forecast_it, num_series=len(test_dataset)
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


class BacktestInformation(NamedTuple):
    train_dataset_stats: DatasetStatistics
    test_dataset_stats: DatasetStatistics
    estimator: Estimator
    agg_metrics: Dict[str, float]

    # @staticmethod
    # def make_from_log(log_file):
    #     with open(log_file, "r") as f:
    #         return BacktestInformation.make_from_log_contents(
    #             "\n".join(f.readlines())
    #         )

    # @staticmethod
    # def make_from_log_contents(log_contents):
    #     messages = dict(re.findall(r"pts\[(.*)\]: (.*)", log_contents))

    #     # avoid to fail if a key is missing for instance in the case a run did
    #     # not finish so that we can still get partial information
    #     try:
    #         return BacktestInformation(
    #             train_dataset_stats=eval(
    #                 messages[train_dataset_stats_key]
    #             ),  # TODO: use load
    #             test_dataset_stats=eval(
    #                 messages[test_dataset_stats_key]
    #             ),  # TODO: use load
    #             estimator=load_code(messages[estimator_key]),
    #             agg_metrics={
    #                 k: load_code(v)
    #                 for k, v in messages.items()
    #                 if k.startswith("metric-") and v != "nan"
    #             },
    #         )
    #     except Exception as error:
    #         logging.error(error)
    #         return None
