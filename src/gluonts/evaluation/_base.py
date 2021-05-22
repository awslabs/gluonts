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
import multiprocessing
import sys
from functools import partial
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

import numpy as np
import pandas as pd

from .metrics import (
    abs_error,
    abs_target_mean,
    abs_target_sum,
    calculate_seasonal_error,
    coverage,
    mape,
    mase,
    mse,
    msis,
    owa,
    quantile_loss,
    smape,
)
from gluonts.gluonts_tqdm import tqdm
from gluonts.model.forecast import Forecast, Quantile


def worker_function(evaluator: "Evaluator", inp: tuple):
    ts, forecast = inp
    return evaluator.get_metrics_per_ts(ts, forecast)


def aggregate_all(
    metric_per_ts: pd.DataFrame, agg_funs: Dict[str, str]
) -> Dict[str, float]:
    """
    No filtering applied.

    Both `nan` and `inf` possible in aggregate metrics.
    """
    return {
        key: metric_per_ts[key].agg(agg, skipna=False)
        for key, agg in agg_funs.items()
    }


def aggregate_no_nan(
    metric_per_ts: pd.DataFrame, agg_funs: Dict[str, str]
) -> Dict[str, float]:
    """
    Filter all `nan` but keep `inf`.

    `nan` is only possible in the aggregate metric if all timeseries
    for a metric resulted in `nan`.
    """
    return {
        key: metric_per_ts[key].agg(agg, skipna=True)
        for key, agg in agg_funs.items()
    }


def aggregate_valid(
    metric_per_ts: pd.DataFrame, agg_funs: Dict[str, str]
) -> Dict[str, Union[float, np.ma.core.MaskedConstant]]:
    """
    Filter all `nan` & `inf` values from `metric_per_ts`.

    If all metrics in a column of `metric_per_ts` are `nan` or `inf` the
    result will be `np.ma.masked` for that column.
    """
    metric_per_ts = metric_per_ts.apply(np.ma.masked_invalid)
    return {
        key: metric_per_ts[key].agg(agg, skipna=True)
        for key, agg in agg_funs.items()
    }


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
        Parameter of the MSIS metric from the M4 competition that
        defines the confidence interval.
        For alpha=0.05 (default) the 95% considered is considered in the metric,
        see https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        for more detail on MSIS
    calculate_owa
        Determines whether the OWA metric should also be calculated,
        which is computationally expensive to evaluate and thus slows
        down the evaluation process considerably.
        By default False.
    custom_eval_fn
        Option to include custom evaluation metrics. Expected input is
        a dictionary with keys specifying the name of the custom metric
        and the values are a list containing three elements.
        First, a callable which takes as input target and forecast and
        returns the evaluation metric.
        Second, a string specifying the aggregation metric across all
        time-series, f.e. "mean", "sum".
        Third, either "mean" or "median" to specify whether mean or median
        forecast should be passed to the custom evaluation function.
        E.g. {"RMSE": [rmse, "mean", "median"]}
    num_workers
        The number of multiprocessing workers that will be used to process
        the data in parallel. Default is multiprocessing.cpu_count().
        Setting it to 0 or None means no multiprocessing.
    chunk_size
        Controls the approximate chunk size each workers handles at a time.
        Default is 32.
    ignore_invalid_values
        Ignore `NaN` and `inf` values in the timeseries when calculating metrics.
    aggregation_strategy:
        Function for aggregating per timeseries metrics.
        Available options are:
        aggregate_valid | aggregate_all | aggregate_no_nan
        The default function is aggregate_no_nan.
    """

    default_quantiles = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

    def __init__(
        self,
        quantiles: Iterable[Union[float, str]] = default_quantiles,
        seasonality: Optional[int] = None,
        alpha: float = 0.05,
        calculate_owa: bool = False,
        custom_eval_fn: Optional[Dict] = None,
        num_workers: Optional[int] = multiprocessing.cpu_count(),
        chunk_size: int = 32,
        aggregation_strategy: Callable = aggregate_no_nan,
        ignore_invalid_values: bool = True,
    ) -> None:
        self.quantiles = tuple(map(Quantile.parse, quantiles))
        self.seasonality = seasonality
        self.alpha = alpha
        self.calculate_owa = calculate_owa
        self.custom_eval_fn = custom_eval_fn
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.aggregation_strategy = aggregation_strategy
        self.ignore_invalid_values = ignore_invalid_values

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
            if self.num_workers and not sys.platform == "win32":
                mp_pool = multiprocessing.Pool(
                    initializer=None, processes=self.num_workers
                )
                rows = mp_pool.map(
                    func=partial(worker_function, self),
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

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None, np.ma.core.MaskedConstant]]:
        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        past_data = np.array(self.extract_past_data(time_series, forecast))

        if self.ignore_invalid_values:
            past_data = np.ma.masked_invalid(past_data)
            pred_target = np.ma.masked_invalid(pred_target)

        try:
            mean_fcst = getattr(forecast, "mean", None)
        except NotImplementedError:
            mean_fcst = None

        median_fcst = forecast.quantile(0.5)
        seasonal_error = calculate_seasonal_error(
            past_data, forecast, self.seasonality
        )

        metrics: Dict[str, Union[float, str, None]] = {
            "item_id": forecast.item_id,
            "MSE": mse(pred_target, mean_fcst)
            if mean_fcst is not None
            else None,
            "abs_error": abs_error(pred_target, median_fcst),
            "abs_target_sum": abs_target_sum(pred_target),
            "abs_target_mean": abs_target_mean(pred_target),
            "seasonal_error": seasonal_error,
            "MASE": mase(pred_target, median_fcst, seasonal_error),
            "MAPE": mape(pred_target, median_fcst),
            "sMAPE": smape(pred_target, median_fcst),
            "OWA": np.nan,  # by default not calculated
        }

        if self.custom_eval_fn is not None:
            for k, (eval_fn, _, fcst_type) in self.custom_eval_fn.items():
                if fcst_type == "mean":
                    if mean_fcst is not None:
                        target_fcst = mean_fcst
                    else:
                        logging.warning(
                            "mean_fcst is None, therefore median_fcst is used."
                        )
                        target_fcst = median_fcst
                else:
                    target_fcst = median_fcst

                try:
                    val = {
                        k: eval_fn(
                            pred_target,
                            target_fcst,
                        )
                    }
                except:
                    val = {k: np.nan}

                metrics.update(val)

        try:
            metrics["MSIS"] = msis(
                pred_target,
                forecast.quantile(self.alpha / 2),
                forecast.quantile(1.0 - self.alpha / 2),
                seasonal_error,
                self.alpha,
            )
        except Exception:
            logging.warning("Could not calculate MSIS metric.")
            metrics["MSIS"] = np.nan

        if self.calculate_owa:
            metrics["OWA"] = owa(
                pred_target,
                median_fcst,
                past_data,
                seasonal_error,
                forecast.start_date,
            )

        for quantile in self.quantiles:
            forecast_quantile = forecast.quantile(quantile.value)

            metrics[quantile.loss_name] = quantile_loss(
                pred_target, forecast_quantile, quantile.value
            )
            metrics[quantile.coverage_name] = coverage(
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

        if self.custom_eval_fn is not None:
            for k, (_, agg_type, _) in self.custom_eval_fn.items():
                agg_funs.update({k: agg_type})

        assert (
            set(metric_per_ts.columns) >= agg_funs.keys()
        ), "Some of the requested item metrics are missing."

        totals = self.aggregation_strategy(
            metric_per_ts=metric_per_ts, agg_funs=agg_funs
        )

        # derived metrics based on previous aggregate metrics
        totals["RMSE"] = np.sqrt(totals["MSE"])
        totals["NRMSE"] = totals["RMSE"] / totals["abs_target_mean"]
        totals["ND"] = totals["abs_error"] / totals["abs_target_sum"]

        for quantile in self.quantiles:
            totals[quantile.weighted_loss_name] = (
                totals[quantile.loss_name] / totals["abs_target_sum"]
            )

        totals["mean_absolute_QuantileLoss"] = np.array(
            [totals[quantile.loss_name] for quantile in self.quantiles]
        ).mean()

        totals["mean_wQuantileLoss"] = np.array(
            [
                totals[quantile.weighted_loss_name]
                for quantile in self.quantiles
            ]
        ).mean()

        totals["MAE_Coverage"] = np.mean(
            [
                np.abs(totals[q.coverage_name] - np.array([q.value]))
                for q in self.quantiles
            ]
        )
        return totals, metric_per_ts


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
        custom_eval_fn: Optional[dict] = None,
        num_workers: Optional[int] = None,
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
        num_workers
            The number of multiprocessing workers that will be used to process
            metric for each dimension of the multivariate forecast.
        """
        super().__init__(
            quantiles=quantiles,
            seasonality=seasonality,
            alpha=alpha,
            custom_eval_fn=custom_eval_fn,
            num_workers=num_workers,
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
