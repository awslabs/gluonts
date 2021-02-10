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


from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions



from gluonts.evaluation import Evaluator
from gluonts.evaluation._base import _worker_init, _worker_fun
default_quantiles = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

def call(
        ts_iterator: Iterable[Union[pd.DataFrame, pd.Series]],
        fcst_iterator: Iterable[Forecast],
        metric,
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
    time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast,metric,
) -> Dict[str, Union[float, str, None]]:
    pred_target = np.array(extract_pred_target(time_series, forecast))
    pred_target = np.ma.masked_invalid(pred_target)
    mean_fcst = forecast.mean
    q = 0.5
    if metric == 'MAPE':
        result =  mape(pred_target, mean_fcst)
    if metric == 'MSE':
        result = mse(pred_target,mean_fcst)
    if metric == 'ABS_ERROR':
        result = abs_error(pred_target,mean_fcst)
    if metric == 'QUANTILE_LOSS':
        result = quantile_loss(pred_target,mean_fcst,q)

    return np.average(result)

def extract_pred_target(
        time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
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

def nan_if_masked(a: Union[float, np.ma.core.MaskedConstant]) -> float:
    return a if a is not np.ma.masked else np.nan

@staticmethod
def mse(target, forecast):
    return np.mean(np.square(target - forecast))

@staticmethod
def mape(target, forecast):

    denominator = np.abs(target)
    flag = denominator

    mape = np.mean(
            (np.abs(target - forecast) * (1 - flag)) / (denominator + flag)
        )
    return mape

@staticmethod
def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
    return nan_if_masked(np.sum(np.abs(target - forecast)))

@staticmethod
def quantile_loss(
        target: np.ndarray, forecast: np.ndarray, q: float
    ) -> float:
    return nan_if_masked(
            2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)))
        )

def evaluation(estimator,transformation,net,dataset_test,metric):
    pre = estimator.create_predictor(transformation, net)
    from gluonts.evaluation.backtest import make_evaluation_predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test,  # test dataset
        predictor=pre,  # predictor
        num_samples=100, )  # number of sample paths we want for evaluation

    forecasts = list(forecast_it)
    tss = list(ts_it)
    result = call(iter(tss), iter(forecasts),metric, num_series=len(dataset_test))
    return result



