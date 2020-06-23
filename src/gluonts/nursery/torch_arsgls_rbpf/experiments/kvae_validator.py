import os
import torch
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from utils.utils import shorten_iter
from typing import (
    Dict,
    Iterable,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from gluonts.model.forecast import Forecast


class NoTQDMEvaluator(Evaluator):
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

        it = zip(ts_iterator, fcst_iterator)
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

        metrics_per_ts = pd.DataFrame(rows, dtype=np.float64)
        return self.get_aggregate_metrics(metrics_per_ts)


class KVAEValidator:
    def __init__(self, log_paths, dataset, forecaster, num_samples, num_series,
                 store_filter_and_forecast_result=False,
                 moving_average_param=0.8):
        super().__init__()
        self.log_paths = log_paths
        self.dataset = dataset
        self.forecaster = forecaster
        self.evaluator = NoTQDMEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            num_workers=0,  # buggy with > 0.
        )
        self.num_samples = num_samples
        self.num_series = num_series
        self.store_filter_and_forecast_result = store_filter_and_forecast_result
        self.ma_param = moving_average_param
        self.loss_moving_average = None

    def __call__(self, epoch, save=True):
        agg_metrics = self.evaluate_forecast()
        self.update_metrics(agg_metrics=agg_metrics)
        if save:
            self.save(epoch=epoch)
        return agg_metrics

    def evaluate_forecast(self):
        with torch.no_grad():
            forecast_it, ts_it = make_evaluation_predictions(
                self.dataset,
                predictor=self.forecaster,
                num_samples=self.num_samples,
            )
            if self.num_series is None:
                num_series = len(self.dataset)
            else:
                num_series = self.num_series
                forecast_it = shorten_iter(forecast_it, num_series)
                ts_it = shorten_iter(ts_it, num_series)
            agg_metrics, item_metrics = self.evaluator(
                ts_it, forecast_it,
                num_series=num_series,
            )
            return agg_metrics

    def update_metrics(self, agg_metrics):
        if not hasattr(self, "agg_metrics"):
            self.agg_metrics = {key: [val] for key, val in agg_metrics.items()}
        else:
            for key, val in agg_metrics.items():
                self.agg_metrics[key].append(val)

    def save(self, epoch):
        np.savez(os.path.join(self.log_paths.metrics, f"{epoch}.npz"),
                 self.agg_metrics)

    def load(self, epoch):
        return np.load(
            os.path.join(self.log_paths.metrics, f"{epoch}.npz")).item()
