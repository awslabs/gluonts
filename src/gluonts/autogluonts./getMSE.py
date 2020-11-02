import logging
import multiprocessing
import sys
from itertools import chain, tee
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

# Third-party imports
import numpy as np
import pandas as pd

from gluonts.gluonts_tqdm import tqdm
#from gluonts.time_feature import get_seasonality

# First-party imports
from gluonts.model.forecast import Forecast, Quantile

from estimator import estimator
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

transformation = estimator.create_transformation()
def mse(net,test):
    predictor = estimator.create_predictor(transformation,net)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test))
    return agg_metrics['MSE']

from gluonts.evaluation import Evaluator
from gluonts.evaluation._base import _worker_init, _worker_fun
default_quantiles = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

class new_eval(Evaluator):
    def __init__(
            self,
            quantiles: Iterable[Union[float, str]] = default_quantiles,
            seasonality: Optional[int] = None,
            alpha: float = 0.05,
            calculate_owa: bool = False,
            num_workers: Optional[int] = None,
            chunk_size: Optional[int] = None,
    ) -> None:
        self.quantiles = tuple(map(Quantile.parse, quantiles))
        self.seasonality = seasonality
        self.alpha = alpha
        self.calculate_owa = calculate_owa
        self.zero_tol = 1e-8

        self.num_workers = (
            num_workers
            if num_workers is not None
            else multiprocessing.cpu_count()
        )
        self.chunk_size = chunk_size if chunk_size is not None else 32

    def __call__(
            self,
            ts_iterator: Iterable[Union[pd.DataFrame, pd.Series]],
            fcst_iterator: Iterable[Forecast],
            num_series: Optional[int] = None,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Compute accuracy metrics by comparing actual data to the forecasts.
        Parameters
        ----------
        ts_iterator
            iterator containing true target on the predicted range
        fcst_iterator
            iterator of forecasts on the predicted range
        num_series
            number of series of the iterator
            (optional, only used for displaying progress)
        Returns
        -------
        dict
            Dictionary of aggregated metrics
        pd.DataFrame
            DataFrame containing per-time-series metrics
        """
        ts_iterator = iter(ts_iterator)
        fcst_iterator = iter(fcst_iterator)

        rows = []

        with tqdm(
                zip(ts_iterator, fcst_iterator),
                total=num_series,
                desc="Running evaluation",
        ) as it, np.errstate(invalid="ignore"):

            for ts, forecast in it:
                rows.append(self.get_metrics_per_ts(ts, forecast))


        return rows

def call(
        ts_iterator: Iterable[Union[pd.DataFrame, pd.Series]],
        fcst_iterator: Iterable[Forecast],
        metric = 'MSE',
        num_series: Optional[int] = None,


) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Compute accuracy metrics by comparing actual data to the forecasts.
    Parameters
    ----------
    ts_iterator
        iterator containing true target on the predicted range
    fcst_iterator
        iterator of forecasts on the predicted range
    num_series
        number of series of the iterator
        (optional, only used for displaying progress)
    Returns
    -------
    dict
        Dictionary of aggregated metrics
    pd.DataFrame
        DataFrame containing per-time-series metrics
    """
    ts_iterator = iter(ts_iterator)
    fcst_iterator = iter(fcst_iterator)

    rows = []

    with tqdm(
            zip(ts_iterator, fcst_iterator),
            total=num_series,
            desc="Running evaluation",
    ) as it, np.errstate(invalid="ignore"):

        for ts, forecast in it:

            rows.append(get_metrics_per_ts(ts, forecast,metric))


    return np.average(rows)

def get_metrics_per_ts(
    time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast, metric = 'MSE'
) -> Dict[str, Union[float, str, None]]:
    pred_target = np.array(extract_pred_target(time_series, forecast))
    pred_target = np.ma.masked_invalid(pred_target)

    mean_fcst = forecast.quantile('p50')

    mean_fcst = forecast.mean
    error =  get_error(pred_target, forecast,metric)
    return error

def extract_pred_target(
        time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast,
) -> np.ndarray:
    """
    Parameters
    ----------
    time_series
    forecast
    Returns
    -------
    np.ndarray
        time series cut in the Forecast object dates
    """
    assert forecast.index.intersection(time_series.index).equals(
        forecast.index
    ), (
        "Cannot extract prediction target since the index of forecast is outside the index of target\n"
        f"Index of forecast: {forecast.index}\n Index of target: {time_series.index}"
    )

    # cut the time series using the dates of the forecast object
    return np.atleast_1d(
        np.squeeze(time_series.loc[forecast.index].transpose())
    )

def get_error(target, forecast,metric):
    if metric == 'MSE':
        forecast_mean = forecast.mean
        return np.mean(np.square(target - forecast_mean))
    if metric == 'abs_error':
        forecast_mean = forecast.mean
        return np.sum(np.abs(target - forecast_mean))
    if metric[0] == 'quantile_loss':
        q = metric[1]
        forecast_q = forecast.quantile(str(q)*100)
        return 2.0 * np.sum(np.abs((forecast_q - target)* ((target <= forecast_q) - metric[1])))







