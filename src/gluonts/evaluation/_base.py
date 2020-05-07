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
import re
import sys

from collections import Sized
from functools import lru_cache
from itertools import chain, tee
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
)

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.model.forecast import Forecast, Quantile
from gluonts.gluonts_tqdm import tqdm


@lru_cache()
def get_seasonality(freq: str) -> int:
    """
    Returns the default seasonality for a given freq str. E.g. for

      2H -> 12

    """
    match = re.match(r"(\d*)(\w+)", freq)
    assert match, "Cannot match freq regex"
    mult, base_freq = match.groups()
    multiple = int(mult) if mult else 1

    seasonalities = {"H": 24, "D": 1, "W": 1, "M": 12, "B": 5}
    if base_freq in seasonalities:
        seasonality = seasonalities[base_freq]
    else:
        seasonality = 1
    if seasonality % multiple != 0:
        logging.warning(
            f"multiple {multiple} does not divide base "
            f"seasonality {seasonality}."
            f"Falling back to seasonality 1"
        )
        return 1
    return seasonality // multiple


class Evaluator:
    """
    Evaluator class, to compute accuracy metrics by comparing observations
    to forecasts.

    Parameters
    ----------
    quantiles
        list of strings of the form 'p10' or floats in [0, 1] with
        the quantile levels
    seasonality
        seasonality to use for seasonal_error, if nothing is passed
        uses the default seasonality
        for the given series frequency as returned by `get_seasonality`
    alpha
        parameter of the MSIS metric from M4 competition that
        defines the confidence interval
        for alpha=0.05 the 95% considered is considered in the metric,
        see https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4
        -Competitors-Guide.pdf for more detail on MSIS
    calculate_owa
        Determines whether the OWA metric should also be calculated,
        which is computationally expensive to evaluate and thus slows
        down the evaluation process considerably.
        By default False.
    num_workers
        The number of multiprocessing workers that will be used to process
        the data in parallel.
        Default is multiprocessing.cpu_count().
        Setting it to 0 means no multiprocessing.
    chunk_size
        Controls the approximate chunk size each workers handles at a time.
        Default is 32.
    """

    default_quantiles = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

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
            if self.num_workers > 0 and not sys.platform == "win32":
                mp_pool = multiprocessing.Pool(
                    initializer=_worker_init(self), processes=self.num_workers
                )
                rows = mp_pool.map(
                    func=_worker_fun,
                    iterable=iter(it),
                    chunksize=self.chunk_size,
                )
                mp_pool.close()
                mp_pool.join()
            else:
                for ts, forecast in it:
                    rows.append(self.get_metrics_per_ts(ts, forecast))

        assert not any(
            True for _ in ts_iterator
        ), "ts_iterator has more elements than fcst_iterator"

        assert not any(
            True for _ in fcst_iterator
        ), "fcst_iterator has more elements than ts_iterator"

        if num_series is not None:
            assert (
                len(rows) == num_series
            ), f"num_series={num_series} did not match number of elements={len(rows)}"

        # If all entries of a target array are NaNs, the resulting metric will have value "masked". Pandas does not
        # handle masked values correctly. Thus we set dtype=np.float64 to convert masked values back to NaNs which
        # are handled correctly by pandas Dataframes during aggregation.
        metrics_per_ts = pd.DataFrame(rows, dtype=np.float64)
        return self.get_aggregate_metrics(metrics_per_ts)

    @staticmethod
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

    # This method is needed for the owa calculation
    # It extracts the training sequence from the Series or DataFrame to a numpy array
    @staticmethod
    def extract_past_data(
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
            time series without the forecast dates
        """

        assert forecast.index.intersection(time_series.index).equals(
            forecast.index
        ), (
            "Index of forecast is outside the index of target\n"
            f"Index of forecast: {forecast.index}\n Index of target: {time_series.index}"
        )

        # Remove the prediction range
        # If the prediction range is not in the end of the time series,
        # everything after the prediction range is truncated
        date_before_forecast = forecast.index[0] - forecast.index[0].freq
        return np.atleast_1d(
            np.squeeze(time_series.loc[:date_before_forecast].transpose())
        )

    def seasonal_error(
        self, past_data: np.ndarray, forecast: Forecast
    ) -> float:
        r"""
        .. math::

            seasonal_error = mean(|Y[t] - Y[t-m]|)

        where m is the seasonal frequency
        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """
        # Check if the length of the time series is larger than the seasonal frequency
        seasonality = (
            self.seasonality
            if self.seasonality
            else get_seasonality(forecast.freq)
        )
        if seasonality < len(past_data):
            forecast_freq = seasonality
        else:
            # edge case: the seasonal freq is larger than the length of ts
            # revert to freq=1
            # logging.info('The seasonal frequency is larger than the length of the time series. Reverting to freq=1.')
            forecast_freq = 1
        y_t = past_data[:-forecast_freq]
        y_tm = past_data[forecast_freq:]

        seasonal_mae = np.mean(abs(y_t - y_tm))

        return seasonal_mae if seasonal_mae is not np.ma.masked else np.nan

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None]]:
        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        pred_target = np.ma.masked_invalid(pred_target)

        # required for seasonal_error and owa calculation
        past_data = np.array(self.extract_past_data(time_series, forecast))
        past_data = np.ma.masked_invalid(past_data)

        try:
            mean_fcst = forecast.mean
        except:
            mean_fcst = None
        median_fcst = forecast.quantile(0.5)
        seasonal_error = self.seasonal_error(past_data, forecast)
        # For MSIS: alpha/2 quantile may not exist. Find the closest.
        lower_q = min(
            self.quantiles, key=lambda q: abs(q.value - self.alpha / 2)
        )
        upper_q = min(
            reversed(self.quantiles),
            key=lambda q: abs(q.value - (1 - self.alpha / 2)),
        )

        metrics = {
            "item_id": forecast.item_id,
            "MSE": self.mse(pred_target, mean_fcst)
            if mean_fcst is not None
            else None,
            "abs_error": self.abs_error(pred_target, median_fcst),
            "abs_target_sum": self.abs_target_sum(pred_target),
            "abs_target_mean": self.abs_target_mean(pred_target),
            "seasonal_error": seasonal_error,
            "MASE": self.mase(pred_target, median_fcst, seasonal_error),
            "MAPE": self.mape(pred_target, median_fcst),
            "sMAPE": self.smape(pred_target, median_fcst),
            "OWA": np.nan,  # by default not calculated
            "MSIS": self.msis(
                pred_target,
                forecast.quantile(lower_q.value),
                forecast.quantile(upper_q.value),
                seasonal_error,
                self.alpha,
            ),
        }

        if self.calculate_owa:
            metrics["OWA"] = self.owa(
                pred_target,
                median_fcst,
                past_data,
                seasonal_error,
                forecast.start_date,
            )

        for quantile in self.quantiles:
            forecast_quantile = forecast.quantile(quantile.value)

            metrics[quantile.loss_name] = self.quantile_loss(
                pred_target, forecast_quantile, quantile.value
            )
            metrics[quantile.coverage_name] = self.coverage(
                pred_target, forecast_quantile
            )

        return metrics

    def get_aggregate_metrics(
        self, metric_per_ts: pd.DataFrame
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        agg_funs = {
            "MSE": "mean",
            "abs_error": "sum",
            "abs_target_sum": "sum",
            "abs_target_mean": "mean",
            "seasonal_error": "mean",
            "MASE": "mean",
            "MAPE": "mean",
            "sMAPE": "mean",
            "OWA": "mean",
            "MSIS": "mean",
        }
        for quantile in self.quantiles:
            agg_funs[quantile.loss_name] = "sum"
            agg_funs[quantile.coverage_name] = "mean"

        assert (
            set(metric_per_ts.columns) >= agg_funs.keys()
        ), "The some of the requested item metrics are missing."

        totals = {
            key: metric_per_ts[key].agg(agg) for key, agg in agg_funs.items()
        }

        # derived metrics based on previous aggregate metrics
        totals["RMSE"] = np.sqrt(totals["MSE"])

        flag = totals["abs_target_mean"] == 0
        totals["NRMSE"] = np.divide(
            totals["RMSE"] * (1 - flag), totals["abs_target_mean"] + flag
        )

        flag = totals["abs_target_sum"] == 0
        totals["ND"] = np.divide(
            totals["abs_error"] * (1 - flag), totals["abs_target_sum"] + flag
        )

        all_qLoss_names = [
            quantile.weighted_loss_name for quantile in self.quantiles
        ]
        for quantile in self.quantiles:
            totals[quantile.weighted_loss_name] = np.divide(
                totals[quantile.loss_name], totals["abs_target_sum"]
            )

        totals["mean_wQuantileLoss"] = np.array(
            [totals[ql] for ql in all_qLoss_names]
        ).mean()

        totals["MAE_Coverage"] = np.mean(
            [
                np.abs(totals[q.coverage_name] - np.array([q.value]))
                for q in self.quantiles
            ]
        )
        return totals, metric_per_ts

    @staticmethod
    def mse(target, forecast):
        return np.mean(np.square(target - forecast))

    @staticmethod
    def abs_error(target, forecast):
        return np.sum(np.abs(target - forecast))

    @staticmethod
    def quantile_loss(target, quantile_forecast, q):
        return 2.0 * np.sum(
            np.abs(
                (quantile_forecast - target)
                * ((target <= quantile_forecast) - q)
            )
        )

    @staticmethod
    def coverage(target, quantile_forecast):
        return np.mean((target < quantile_forecast))

    @staticmethod
    def mase(target, forecast, seasonal_error):
        r"""
        .. math::

            mase = mean(|Y - Y_hat|) / seasonal_error

        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """
        flag = seasonal_error == 0
        return (np.mean(np.abs(target - forecast)) * (1 - flag)) / (
            seasonal_error + flag
        )

    @staticmethod
    def mape(target, forecast):
        r"""
        .. math::

            mape = mean(|Y - Y_hat| / |Y|))
        """

        denominator = np.abs(target)
        flag = denominator == 0

        mape = np.mean(
            (np.abs(target - forecast) * (1 - flag)) / (denominator + flag)
        )
        return mape

    @staticmethod
    def smape(target, forecast):
        r"""
        .. math::

            smape = mean(2 * |Y - Y_hat| / (|Y| + |Y_hat|))

        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """

        denominator = np.abs(target) + np.abs(forecast)
        flag = denominator == 0

        smape = 2 * np.mean(
            (np.abs(target - forecast) * (1 - flag)) / (denominator + flag)
        )
        return smape

    @staticmethod
    def owa(
        target: np.ndarray,
        forecast: np.ndarray,
        past_data: np.ndarray,
        seasonal_error: float,
        start_date: pd.Timestamp,
    ) -> float:
        r"""
        .. math::

            owa = 0.5*(smape/smape_naive + mase/mase_naive)

        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """
        # avoid import error due to circular dependency
        from gluonts.model.naive_2 import naive_2

        # calculate the forecast of the seasonal naive predictor
        naive_median_fcst = naive_2(
            past_data, len(target), freq=start_date.freqstr
        )

        owa = 0.5 * (
            (
                Evaluator.smape(target, forecast)
                / Evaluator.smape(target, naive_median_fcst)
            )
            + (
                Evaluator.mase(target, forecast, seasonal_error)
                / Evaluator.mase(target, naive_median_fcst, seasonal_error)
            )
        )

        return owa

    @staticmethod
    def msis(target, lower_quantile, upper_quantile, seasonal_error, alpha):
        r"""
        :math:

            msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U]) /seasonal_error

        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """
        numerator = np.mean(
            upper_quantile
            - lower_quantile
            + 2.0
            / alpha
            * (lower_quantile - target)
            * (target < lower_quantile)
            + 2.0
            / alpha
            * (target - upper_quantile)
            * (target > upper_quantile)
        )

        flag = seasonal_error == 0
        return (numerator * (1 - flag)) / (seasonal_error + flag)

    @staticmethod
    def abs_target_sum(target):
        return np.sum(np.abs(target))

    @staticmethod
    def abs_target_mean(target):
        return np.mean(np.abs(target))


class MultivariateEvaluator(Evaluator):
    """
    
    The MultivariateEvaluator class owns functionality for evaluating
    multidimensional target arrays of shape
    (target_dimensionality, prediction_length).

    Evaluations of individual dimensions will be stored with the corresponding
    dimension prefix and contain the metrics calculated by only this dimension.
    Metrics with the plain metric name correspond to metrics calculated over
    all dimensions.
    Additionally, the user can provide additional aggregation functions that
    first aggregate the target and forecast over dimensions and then calculate
    the metric. These metrics will be prefixed with m_<aggregation_fun_name>_

    The evaluation dimensions can be set by the user.

    Example:
        {'0_MSE': 0.004307240342677687, # MSE of dimension 0
        '0_abs_error': 1.6246897801756859,
        '1_MSE': 0.003949341769475723, # MSE of dimension 1
        '1_abs_error': 1.5052175521850586,
        'MSE': 0.004128291056076705, # MSE of all dimensions
        'abs_error': 3.1299073323607445,
        'm_sum_MSE': 0.02 # MSE of aggregated target and aggregated forecast
        (if target_agg_funcs is set).
        'm_sum_abs_error': 4.2}
    """

    def __init__(
        self,
        quantiles: Iterable[Union[float, str]] = np.linspace(0.1, 0.9, 9),
        seasonality: Optional[int] = None,
        alpha: float = 0.05,
        eval_dims: List[int] = None,
        target_agg_funcs: Dict[str, Callable] = {},
    ) -> None:
        """

        Parameters
        ----------
        quantiles
            list of strings of the form 'p10' or floats in [0, 1] with the
            quantile levels
        seasonality
            seasonality to use for seasonal_error, if nothing is passed uses
            the default seasonality for the given series frequency as
            returned by `get_seasonality`
        alpha
            parameter of the MSIS metric that defines the CI,
            e.g., for alpha=0.05 the 95% CI is considered in the metric.
        eval_dims
            dimensions of the target that will be evaluated.
        target_agg_funcs
            pass key-value pairs that define aggregation functions over the
            dimension axis. Useful to compute metrics over aggregated target
            and forecast (typically sum or mean).
        """
        super().__init__(
            quantiles=quantiles, seasonality=seasonality, alpha=alpha
        )
        self._eval_dims = eval_dims
        self.target_agg_funcs = target_agg_funcs

    @staticmethod
    def extract_target_by_dim(
        it_iterator: Iterator[pd.DataFrame], dim: int
    ) -> Iterator[pd.DataFrame]:
        for i in it_iterator:
            yield (i[dim])

    @staticmethod
    def extract_forecast_by_dim(
        forecast_iterator: Iterator[Forecast], dim: int
    ) -> Iterator[Forecast]:
        for forecast in forecast_iterator:
            yield forecast.copy_dim(dim)

    @staticmethod
    def extract_aggregate_target(
        it_iterator: Iterator[pd.DataFrame], agg_fun: Callable
    ) -> Iterator[pd.DataFrame]:
        for i in it_iterator:
            yield i.agg(agg_fun, axis=1)

    @staticmethod
    def extract_aggregate_forecast(
        forecast_iterator: Iterator[Forecast], agg_fun: Callable
    ) -> Iterator[Forecast]:
        for forecast in forecast_iterator:
            yield forecast.copy_aggregate(agg_fun)

    @staticmethod
    def peek(iterator: Iterator[Any]) -> Tuple[Any, Iterator[Any]]:
        peeked_object = iterator.__next__()
        iterator = chain([peeked_object], iterator)
        return peeked_object, iterator

    @staticmethod
    def get_target_dimensionality(forecast: Forecast) -> int:
        target_dim = forecast.dim()
        assert target_dim > 1, (
            f"the dimensionality of the forecast should be larger than 1, "
            f"but got {target_dim}. "
            f"Please use the Evaluator to evaluate 1D forecasts."
        )
        return target_dim

    def get_eval_dims(self, target_dimensionality: int) -> List[int]:
        eval_dims = (
            self._eval_dims
            if self._eval_dims is not None
            else list(range(0, target_dimensionality))
        )
        assert max(eval_dims) < target_dimensionality, (
            f"eval dims should range from 0 to target_dimensionality - 1, "
            f"but got max eval_dim {max(eval_dims)}"
        )
        return eval_dims

    def calculate_aggregate_multivariate_metrics(
        self,
        ts_iterator: Iterator[pd.DataFrame],
        forecast_iterator: Iterator[Forecast],
        agg_fun: Callable,
    ) -> Dict[str, float]:
        """

        Parameters
        ----------
        ts_iterator
            Iterator over time series
        forecast_iterator
            Iterator over forecasts
        agg_fun
            aggregation function
        Returns
        -------
        Dict[str, float]
            dictionary with aggregate datasets metrics
        """
        agg_metrics, _ = super(MultivariateEvaluator, self).__call__(
            self.extract_aggregate_target(ts_iterator, agg_fun),
            self.extract_aggregate_forecast(forecast_iterator, agg_fun),
        )
        return agg_metrics

    def calculate_aggregate_vector_metrics(
        self,
        all_agg_metrics: Dict[str, float],
        all_metrics_per_ts: pd.DataFrame,
    ) -> Dict[str, float]:
        """

        Parameters
        ----------
        all_agg_metrics
            dictionary with aggregate metrics of individual dimensions
        all_metrics_per_ts
            DataFrame containing metrics for all time series of all evaluated
            dimensions

        Returns
        -------
        Dict[str, float]
            dictionary with aggregate metrics (of individual (evaluated)
            dimensions and the entire vector)
        """
        vector_aggregate_metrics, _ = self.get_aggregate_metrics(
            all_metrics_per_ts
        )
        for key, value in vector_aggregate_metrics.items():
            all_agg_metrics[key] = value
        return all_agg_metrics

    def __call__(
        self,
        ts_iterator: Iterable[pd.DataFrame],
        fcst_iterator: Iterable[Forecast],
        num_series=None,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        ts_iterator = iter(ts_iterator)
        fcst_iterator = iter(fcst_iterator)

        all_agg_metrics = dict()
        all_metrics_per_ts = list()

        peeked_forecast, fcst_iterator = self.peek(fcst_iterator)
        target_dimensionality = self.get_target_dimensionality(peeked_forecast)
        eval_dims = self.get_eval_dims(target_dimensionality)

        ts_iterator_set = tee(
            ts_iterator, target_dimensionality + len(self.target_agg_funcs)
        )
        fcst_iterator_set = tee(
            fcst_iterator, target_dimensionality + len(self.target_agg_funcs)
        )

        for dim in eval_dims:
            agg_metrics, metrics_per_ts = super(
                MultivariateEvaluator, self
            ).__call__(
                self.extract_target_by_dim(ts_iterator_set[dim], dim),
                self.extract_forecast_by_dim(fcst_iterator_set[dim], dim),
            )

            all_metrics_per_ts.append(metrics_per_ts)

            for metric, value in agg_metrics.items():
                all_agg_metrics[f"{dim}_{metric}"] = value

        all_metrics_per_ts = pd.concat(all_metrics_per_ts)
        all_agg_metrics = self.calculate_aggregate_vector_metrics(
            all_agg_metrics, all_metrics_per_ts
        )

        if self.target_agg_funcs:
            multivariate_metrics = {
                agg_fun_name: self.calculate_aggregate_multivariate_metrics(
                    ts_iterator_set[-(index + 1)],
                    fcst_iterator_set[-(index + 1)],
                    agg_fun,
                )
                for index, (agg_fun_name, agg_fun) in enumerate(
                    self.target_agg_funcs.items()
                )
            }

            for key, metric_dict in multivariate_metrics.items():
                prefix = f"m_{key}_"
                for metric, value in metric_dict.items():
                    all_agg_metrics[prefix + metric] = value

        return all_agg_metrics, all_metrics_per_ts


# This is required for the multiprocessing to work.
_worker_evaluator: Optional[Evaluator] = None


def _worker_init(evaluator: Evaluator):
    global _worker_evaluator
    _worker_evaluator = evaluator


def _worker_fun(inp: tuple):
    ts, forecast = inp
    global _worker_evaluator
    assert isinstance(
        _worker_evaluator, Evaluator
    ), "Something went wrong with the worker initialization."
    return _worker_evaluator.get_metrics_per_ts(ts, forecast)
