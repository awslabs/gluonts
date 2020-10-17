
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


# First-party imports
from gluonts.model.forecast import Forecast, Quantile

default_quantiles = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9



def get_result(
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
        print(forecast_mean,target)
        return np.mean(np.square(target - forecast_mean))
    if metric == 'abs_error':
        forecast_mean = forecast.mean
        return np.sum(np.abs(target - forecast_mean))
    if metric[0] == 'quantile_loss':
        q = metric[1]
        forecast_q = forecast.quantile(str(q)*100)
        return 2.0 * np.sum(np.abs((forecast_q - target)* ((target <= forecast_q) - metric[1])))







