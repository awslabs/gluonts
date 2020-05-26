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
import re
from typing import Dict, Iterator, NamedTuple, Optional, Tuple, Union

# Third-party imports
import pandas as pd

# First-party imports
import gluonts  # noqa
from gluonts import transform
from gluonts.core.component import get_mxnet_context
from gluonts.core.serde import load_code
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.dataset.stat import (
    DatasetStatistics,
    calculate_dataset_statistics,
)
from gluonts.evaluation import Evaluator
from gluonts.model.estimator import Estimator, GluonEstimator
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import GluonPredictor, Predictor
from gluonts.support.util import maybe_len
from gluonts.transform import TransformedDataset
from mxnet.ndarray import NDArray

def generate_rolling_datasets(
    dataset: Dataset,
    window_size: int,
    start_time: pd.Timestamp, 
    end_time: pd.Timestamp
    ) -> dict:
    """
    Returns a dictionary containing two dataset generators for using when performing rolling origin evaluations.
    dict['to_evaluate'] contains the data that predictions are to be compared with. 
    dict['to_predict'] contains the data that the predictor will use to generate forecasts. The target values of this dataset is of varying lengths depending on the provided window size, start_time and end_time parameters.
    
    in case of a window size which is not a multiple of the range between the start_time and the end_time will have a shorter window equal to the remainder.

    Any target values after the end_time is removed from both the 'to_evaluate' and the 'to_predict' datasets.

    Parameters
    ----------
    dataset
        Dataset to generate the rolling forecasting datasets from
    window_size
        Int which represents the amount of items in a timeseries should be forecasted and evaluated on. The prediction length of the predictor to use is suitable as window_size.
    start_time
        The start of the time period which rolling forecasts should be applied
    end_time
        The end of the time period which rolling forecasts should be applied

    Returns
    -------
    dict 
        Contains the dataset to provide to the predictor and the dataset to provide to the evaluator
    """

    assert(dataset, start_time, end_time, window_size)
    
    _freq = start_time.freq
    _num_items_in_rolling = len(pd.date_range(start=start_time, end=end_time, freq=_freq))

    # ensure that the window size is small enough to allow rolling
    assert(_num_items_in_rolling > window_size)

    _remainder = _num_items_in_rolling % window_size
    _remainder_date = end_time - _freq*_remainder
    _num_window_iterations = int((_num_items_in_rolling - _remainder)/window_size)
    if _remainder:
        _num_window_iterations = _num_window_iterations + 1

    # TODO test this
    def remove_remainder(data):
        data = data.copy()
        start = data['start']
        target = data['target']
        end_date = start + _freq*(len(target)-1)
        
        # delete all values after the remainder date to allow these to be evaluated
        if end_date > _remainder_date:
            length_to_remainder = len(pd.date_range(start=start, end=_remainder_date, freq=_freq))
            data['target'] = target[:length_to_remainder]
        return data

    # removes target values appearing after end_time
    def truncate_end(data):
        data = data.copy()
        data['rolling'] = (start_time, end_time)

        #calc number datapoints needed from start of timeseries until end of rolling test range
        timerange = pd.date_range(
            start=data['start'], 
            end=end_time, 
            freq=_freq
        )
        max_items = len(timerange)

        # keep values until end of test range
        if max_items < len(data['target']):
            data['target'] = data['target'][:max_items]
        
        return data
    
    # generator to create rolling datasets
    def perform_roll(dataset):
        for timeseries in dataset:
            windows_todo = _num_window_iterations
            window_start_date = end_time
            while windows_todo:
                _ts = timeseries.copy()
                
                # 1. calc length of target 
                target_length = len(_ts['target'])

                # 2. calc new length of target
                window_start_date = window_start_date - _freq * window_size
                length_to_window = len(pd.date_range(start=_ts['start'], end=window_start_date, freq=_freq))
                
                # 3. if length of target < new length of target set flag "not touched" 
                if target_length < length_to_window:
                    # this avoids duplicate evaluations in the evaluator
                    _ts['not_touched'] = True
                else:
                    _ts['target'] = _ts['target'][:length_to_window]
                
                # 4. yield the truncated/rolled timeseries even if it is "not touched" 
                windows_todo = windows_todo - 1
                yield _ts

    # creates a generator for a evaluation dataset of the same 
    # number timeseries as the rolling set
    def generate_test(dataset):
        for timeseries in dataset:
            ts = timeseries.copy()
            for _ in range(_num_window_iterations):
                yield ts
            
    # this will be the new dataset used for evaluating the rolled values
    dataset_truncated_end = TransformedDataset(
            dataset, transformations=[
                transform.AdhocTransform(truncate_end),
            ]
        )
    dataset_eval = generate_test(dataset_truncated_end)

    # this dataset contains all timeseries with all dates after the _remainder_date being removed 
    # thus these values will be forecasted on
    dataset_rolled = TransformedDataset(
            dataset_truncated_end, transformations=[
                transform.AdhocTransform(truncate_end),
                transform.AdhocTransform(remove_remainder)
            ]
        )
    dataset_rolled = perform_roll(dataset)

    # we now have the shortened test dataset and we need to generate the rolls
    #dataset_rolled = TransformedDataset(
    #        dataset_cut_remainder, transformations=[transform.AdhocTransform(perform_roll)]
    #    )
    return {'to_predict':dataset_rolled, 'to_evaluate':dataset_eval}


def make_evaluation_predictions(
    dataset: Dataset, 
    predictor: Predictor, 
    num_samples: int,
    rolling_time_range: Optional[tuple] = None #(pd.Timestamp, pd.Timestamp) # is of size prediction_length * N where N is the amount of windows to use
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
    rolling_time_range
        Optional parameter which, when set, causes rolling forecasting to be used when predicting and evaluating accuracy. Should have the format (start_time: pd.Timestamp, end_time: pd.Timestamp) and should describe the time range on which rolling forecasts should be applied
    Returns
    -------
    """

    prediction_length = predictor.prediction_length
    freq = predictor.freq
    lead_time = predictor.lead_time

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
        data["target"] = target[..., : -prediction_length - lead_time]
        return data

    # TODO filter out time series with target shorter than prediction length
    # TODO or fix the evaluator so it supports missing values instead (all
    # TODO the test set may be gone otherwise with such a filtering)

    if rolling_time_range: #TODO Ensure this works
        generate_rolling_datasets(dataset, prediction_length, *rolling_time_range)
    else:
        dataset_trunc = TransformedDataset(
            dataset, transformations=[transform.AdhocTransform(truncate_target)]
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
    logger.info(f"gluonts[{message}]: {variable}")


def backtest_metrics(
    train_dataset: Optional[Dataset],
    test_dataset: Dataset,
    forecaster: Union[Estimator, Predictor],
    evaluator=Evaluator(
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ),
    num_samples: int = 100,
    logging_file: Optional[str] = None,
    use_symbol_block_predictor: Optional[bool] = False,
    num_workers: Optional[int] = None,
    num_prefetch: Optional[int] = None,
    **kwargs,
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
    use_symbol_block_predictor
        Use a :class:`SymbolBlockPredictor` during testing.
    num_workers
        The number of multiprocessing workers to use for data preprocessing.
        By default 0, in which case no multiprocessing will be utilized.
    num_prefetch
        The number of prefetching batches only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain batches before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more shared_memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_workers` in this case.
        By default it defaults to `num_workers * 2`.

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

        if isinstance(forecaster, GluonEstimator) and isinstance(
            predictor, GluonPredictor
        ):
            inference_data_loader = InferenceDataLoader(
                dataset=test_dataset,
                transform=predictor.input_transform,
                batch_size=forecaster.trainer.batch_size,
                ctx=forecaster.trainer.ctx,
                dtype=forecaster.dtype,
                num_workers=num_workers,
                num_prefetch=num_prefetch,
                **kwargs,
            )

            if forecaster.trainer.hybridize:
                predictor.hybridize(batch=next(iter(inference_data_loader)))

            if use_symbol_block_predictor:
                predictor = predictor.as_symbol_block_predictor(
                    batch=next(iter(inference_data_loader))
                )
    else:
        predictor = forecaster

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


class BacktestInformation(NamedTuple):
    train_dataset_stats: DatasetStatistics
    test_dataset_stats: DatasetStatistics
    estimator: Estimator
    agg_metrics: Dict[str, float]

    @staticmethod
    def make_from_log(log_file):
        with open(log_file, "r") as f:
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
