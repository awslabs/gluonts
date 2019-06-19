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
from functools import lru_cache
from itertools import chain, tee
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

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
    match = re.match(r'(\d*)(\w+)', freq)
    assert match, "Cannot match freq regex"
    multiple, base_freq = match.groups()
    multiple = int(multiple) if multiple else 1

    seasonalities = {'H': 24, 'D': 1, 'W': 1, 'M': 12, 'B': 5}
    if base_freq in seasonalities:
        seasonality = seasonalities[base_freq]
    else:
        seasonality = 1
    if seasonality % multiple != 0:
        logging.warning(
            f'multiple {multiple} does not divide base seasonality {seasonality}.'
            f'Falling back to seasonality 1'
        )
        return 1
    return seasonality // multiple


class Evaluator:
    def __init__(
        self,
        quantiles: Iterable[Union[float, str]] = [f"0.{i}" for i in range(10)],
        seasonality: Optional[int] = None,
        alpha: float = 0.05,
    ):
        """

        Parameters
        ----------
        quantiles
            list of strings of the form 'p10' or floats in [0, 1] with the quantile levels
        seasonality
            seasonality to use for seasonal_error, if nothing is passed uses the default seasonality
            for the given series frequency as returned by `get_seasonality`
        alpha
            parameter of the MSIS metric from M4 competition that defines the confidence interval
            for alpha=0.05 the 95% considered is considered in the metric,
            see https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf for more detail on MSIS
        """

        self.quantile_values, self.quantile_names = zip(
            *map(Quantile.parse, quantiles)
        )
        self.seasonality = seasonality
        self.alpha = alpha

    def __call__(
        self,
        ts_iterator: Iterable[Union[pd.DataFrame, pd.Series]],
        fcst_iterator: Iterable[Forecast],
        num_series: Optional[int] = None,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Parameters
        ----------
        ts_iterator
            iterator containing true target on the predicted range
        fcst_iterator
            iterator of forecasts on the predicted range
        num_series
            number of series of the iterator (optional only used for displaying progress)

        Returns
        -------
        dict
            Dictionary of aggregated metrics and dataframe containing per-time-series metrics
        """
        ts_iterator = iter(ts_iterator)
        fcst_iterator = iter(fcst_iterator)

        rows = []

        with tqdm(
            zip(ts_iterator, fcst_iterator),
            total=num_series,
            desc='Running evaluation',
        ) as it:
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
            ), f'num_series={num_series} did not match number of elements={len(rows)}'

        # If all entries of a target array are NaNs, the resulting metric will have value "masked". Pandas does not
        # handle masked values correctly. Thus we set dtype=np.float64 to convert masked values back to NaNs which
        # are handled correctly by pandas Dataframes during aggregation.
        metrics_per_ts = pd.DataFrame(rows, dtype=np.float64)
        return self.get_aggregate_metrics(metrics_per_ts)

    @staticmethod
    def extract_pred_target(
        time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Union[pd.Series, pd.DataFrame]:
        """

        Parameters
        ----------
        time_series
        forecast

        Returns
        -------
        Union[pandas.Series, pandas.DataFrame]
            time series cut in the Forecast object dates
        """
        assert forecast.index.intersection(time_series.index).equals(
            forecast.index
        ), (
            "Cannot extract prediction target since the index of forecast is outside the index of target\n"
            f"Index of forecast: {forecast.index}\n Index of target: {time_series.index}"
        )

        # cut the time series using the dates of the forecast object
        return np.squeeze(time_series.loc[forecast.index].transpose())

    def seasonal_error(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> float:
        """
        seasonal_error = mean(|Y[t] - Y[t-m]|)
        where m is the seasonal frequency
        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """
        # Remove the prediction range
        # If the prediction range is not in the end of the time series,
        # everything after the prediction range is truncated
        forecast_date = pd.Timestamp(forecast.start_date, freq=forecast.freq)
        date_before_forecast = forecast_date - 1 * forecast_date.freq
        ts = time_series[:date_before_forecast]

        # Check if the length of the time series is larger than the seasonal frequency
        seasonality = (
            self.seasonality
            if self.seasonality
            else get_seasonality(forecast.freq)
        )
        if seasonality < len(ts):
            forecast_freq = seasonality
        else:
            # edge case: the seasonal freq is larger than the length of ts
            # revert to freq=1
            # logging.info('The seasonal frequency is larger than the length of the time series. Reverting to freq=1.')
            forecast_freq = 1
        y_t = np.ma.masked_invalid(ts.values[:-forecast_freq])
        y_tm = np.ma.masked_invalid(ts.values[forecast_freq:])

        return float(np.mean(abs(y_t - y_tm)))

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, float]:
        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        pred_target = np.ma.masked_invalid(pred_target)

        mean_fcst = forecast.mean
        median_fcst = forecast.quantile(0.5)
        seasonal_error = self.seasonal_error(time_series, forecast)
        # For MSIS: alpha/2 quantile may not exist. Find the closest.
        lower_q = min(
            self.quantile_values, key=lambda x: abs(x - self.alpha / 2)
        )
        upper_q = min(
            list(reversed(self.quantile_values)),
            key=lambda x: abs(x - (1 - self.alpha / 2)),
        )

        metrics = {
            "item_id": forecast.item_id,
            "MSE": self.mse(pred_target, mean_fcst),
            "abs_error": self.abs_error(pred_target, median_fcst),
            "abs_target_sum": self.abs_target_sum(pred_target),
            "abs_target_mean": self.abs_target_mean(pred_target),
            "seasonal_error": seasonal_error,
            "MASE": self.mase(pred_target, median_fcst, seasonal_error),
            'sMAPE': self.smape(pred_target, median_fcst),
            'MSIS': self.msis(
                pred_target,
                forecast.quantile(lower_q),
                forecast.quantile(upper_q),
                seasonal_error,
                self.alpha,
            ),
        }

        for q, q_name in zip(self.quantile_values, self.quantile_names):
            m = 'QuantileLoss[{}]'.format(q_name)
            metrics[m] = self.quantile_loss(
                pred_target, forecast.quantile(q), q
            )

            m = 'Coverage[{}]'.format(q_name)
            metrics[m] = self.coverage(pred_target, forecast.quantile(q))

        return metrics

    def get_aggregate_metrics(
        self, metric_per_ts: pd.DataFrame
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        agg_funs = {
            "MSE": 'mean',
            "abs_error": 'sum',
            "abs_target_sum": 'sum',
            "abs_target_mean": 'mean',
            "seasonal_error": 'mean',
            "MASE": 'mean',
            'sMAPE': 'mean',
            'MSIS': 'mean',
        }
        for q, q_name in zip(self.quantile_values, self.quantile_names):
            agg_funs['QuantileLoss[{}]'.format(q_name)] = 'sum'
            agg_funs['Coverage[{}]'.format(q_name)] = 'mean'

        assert set(metric_per_ts.columns).issuperset(
            agg_funs.keys()
        ), 'The some of the requested item metrics are missing.'

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

        all_qLoss_names = []
        for q, q_name in zip(self.quantile_values, self.quantile_names):
            qLoss_name = "wQuantileLoss[{}]".format(q_name)
            all_qLoss_names.append(qLoss_name)
            totals[qLoss_name] = np.divide(
                totals['QuantileLoss[{}]'.format(q_name)],
                totals["abs_target_sum"],
            )

        totals['mean_wQuantileLoss'] = np.array(
            [totals[ql] for ql in all_qLoss_names]
        ).mean()

        totals['MAE_Coverage'] = np.array(
            [
                np.abs(totals['Coverage[{}]'.format(q)] - np.array([q]))
                for q in self.quantile_values
            ]
        ).mean()
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
        """
        mase = mean(|Y - Y_hat|) / seasonal_error
        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """
        flag = seasonal_error == 0
        return (np.mean(np.abs(target - forecast)) * (1 - flag)) / (
            seasonal_error + flag
        )

    @staticmethod
    def smape(target, forcecast):
        """
        smape = mean(2 * |Y - Y_hat| / (|Y| + |Y_hat|))
        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """

        denominator = np.abs(target) + np.abs(forcecast)
        flag = denominator == 0

        smape = 2 * np.mean(
            (np.abs(target - forcecast) * (1 - flag)) / (denominator + flag)
        )
        return smape

    @staticmethod
    def msis(target, lower_quantile, upper_quantile, seasonal_error, alpha):
        """
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
    The MultivariateEvaluator class owns functionality for evaluating multidimensional target arrays of shape
    (target_dimensionality, prediction_length).

    The aggregate metric keys in the output dictionary correspond to the aggregated metrics over the entire target
    array. Additionally, evaluations of individual dimensions will be stored in the corresponding dimension key and
    contain the metrics calculated by only this dimension. Evaluation dimensions can be set by the user.

    Example:
        {0: {'MSE': 0.004307240342677687, 'abs_error': 1.6246897801756859, 'abs_target_sum': 90.0, ...},
        1: {'MSE': 0.003949341769475723, 'abs_error': 1.5052175521850586, 'abs_target_sum': 290.0,...},
        MSE': 0.004128291056076705, 'abs_error': 3.1299073323607445, 'abs_target_sum': 380.0, ...}

    """

    def __init__(
        self,
        quantiles: Iterable[Union[float, str]],
        seasonality: Optional[int] = None,
        alpha: float = 0.05,
        eval_dims: List[int] = None,
    ):
        """

        Parameters
        ----------
        quantiles
            list of strings of the form 'p10' or floats in [0, 1] with the quantile levels
        seasonality
            seasonality to use for seasonal_error, if nothing is passed uses the default seasonality
            for the given series frequency as returned by `get_seasonality`
        alpha
            parameter of the MSIS metric that defines the CI,
            e.g., for alpha=0.05 the 95% CI is considered in the metric.
        eval_dims
            dimensions of the target that will be evaluated
        """
        super().__init__(
            quantiles=quantiles, seasonality=seasonality, alpha=alpha
        )
        self._eval_dims = eval_dims

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
    def peek(iterator: Iterator[Any]) -> Tuple[Any, Iterator[Any]]:
        peeked_object = iterator.__next__()
        iterator = chain([peeked_object], iterator)
        return peeked_object, iterator

    @staticmethod
    def get_target_dimensionality(forecast: Forecast) -> int:
        target_dim = forecast.dim()
        assert target_dim > 1, (
            f'the dimensionality of the forecast should be larger than 1, but got {target_dim}. '
            f'Please use the Evaluator to evaluate 1D forecasts.'
        )
        return target_dim

    def get_eval_dims(self, target_dimensionality: int) -> List[int]:
        eval_dims = (
            self._eval_dims
            if self._eval_dims is not None
            else list(range(0, target_dimensionality))
        )
        assert (
            max(eval_dims) < target_dimensionality
        ), f'eval dims should range from 0 to target_dimensionality - 1, but got max eval_dim {max(eval_dims)}'
        return eval_dims

    def calculate_aggregate_vector_metrics(
        self,
        all_agg_metrics: Dict[int, Dict[str, float]],
        all_metrics_per_ts: pd.DataFrame,
    ):
        """

        Parameters
        ----------
        all_agg_metrics
            dictionary with aggregate metrics of individual dimensions
        all_metrics_per_ts
            DataFrame containing metrics for all time series of all evaluated dimensions

        Returns
        -------
        Dict[int, Dict[str, float]]
            dictionary with aggregate metrics (of individual (evaluated) dimensions and the entire vector)
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
    ):
        ts_iterator = iter(ts_iterator)
        fcst_iterator = iter(fcst_iterator)

        all_agg_metrics = dict()
        all_metrics_per_ts = list()

        peeked_forecast, fcst_iterator = self.peek(fcst_iterator)
        target_dimensionality = self.get_target_dimensionality(peeked_forecast)
        eval_dims = self.get_eval_dims(target_dimensionality)

        ts_iterator_set = tee(ts_iterator, target_dimensionality)
        fcst_iterator_set = tee(fcst_iterator, target_dimensionality)

        for dim in eval_dims:
            agg_metrics, metrics_per_ts = super(
                MultivariateEvaluator, self
            ).__call__(
                self.extract_target_by_dim(ts_iterator_set[dim], dim),
                self.extract_forecast_by_dim(fcst_iterator_set[dim], dim),
            )

            all_metrics_per_ts.append(metrics_per_ts)
            all_agg_metrics[dim] = agg_metrics

        all_metrics_per_ts = pd.concat(all_metrics_per_ts)
        all_agg_metrics = self.calculate_aggregate_vector_metrics(
            all_agg_metrics, all_metrics_per_ts
        )

        return all_agg_metrics, all_metrics_per_ts
