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


class FilterAndForecastEvaluator(Evaluator):
    def __init__(self, *args, num_workers=0, **kwargs):
        assert num_workers == 0
        super().__init__(*args, **kwargs, num_workers=0)

    def __call__(
            self,
            ts_iterator: Iterable[Union[pd.DataFrame, pd.Series]],
            pred_iterator: Iterable[Forecast],
            num_series: Optional[int] = None,
    ) -> Dict[torch.Tensor, Tuple[Dict[str, float], pd.DataFrame]]:
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
        pred_iterator = iter(pred_iterator)

        rows = []
        losses = []
        forecasts = []
        mpys_filter = []
        Vpys_filter = []

        it = zip(ts_iterator, pred_iterator)
        for ts, fcst_and_loss in it:
            forecast = fcst_and_loss["forecast"]
            filter_loss_timewise = fcst_and_loss["loss"]
            assert filter_loss_timewise.ndim == 1
            losses.append(filter_loss_timewise.sum())
            forecasts.append(forecast.samples)
            # mpys_filter.append(fcst_and_loss["mpy_filter"])
            # Vpys_filter.append(fcst_and_loss["Vpy_filter"])
            rows.append(
                self.get_metrics_per_ts(ts, forecast))  # forecasts metrics

        assert not any(
            True for _ in ts_iterator
        ), "ts_iterator has more elements than fcst_iterator"

        assert not any(
            True for _ in pred_iterator
        ), "fcst_iterator has more elements than ts_iterator"

        if num_series is not None:
            assert (
                    len(rows) == num_series
            ), f"num_series={num_series} did not match number of elements={len(rows)}"

        # If all entries of a target array are NaNs, the resulting metric will have value "masked". Pandas does not
        # handle masked values correctly. Thus we set dtype=np.float64 to convert masked values back to NaNs which
        # are handled correctly by pandas Dataframes during aggregation.
        metrics_per_ts = pd.DataFrame(rows, dtype=np.float64)
        return {
            "metrics": self.get_aggregate_metrics(metrics_per_ts),
            "loss": sum(losses),
            "forecast": np.stack(forecasts, axis=0).transpose((2, 1, 0)),
            # BPT -> TPB
            # "mpy_filter": np.stack(mpys_filter, axis=0).transpose((2, 1, 0, 3)),
            # "Vpy_filter": np.stack(Vpys_filter, axis=0).transpose((2, 1, 0, 3, 4)),
        }


class Validator:
    def __init__(self, log_paths, dataset, forecaster, num_samples, num_series,
                 store_filter_and_forecast_result=False,
                 moving_average_param=0.8):
        super().__init__()
        self.log_paths = log_paths
        self.dataset = dataset
        self.forecaster = forecaster
        self.evaluator = FilterAndForecastEvaluator(
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
            eval_outputs = self.evaluator(
                ts_it, forecast_it,
                num_series=num_series,
            )
            agg_metrics, item_metrics = eval_outputs["metrics"]
            loss = eval_outputs["loss"]
            # T, P, B = eval_outputs["mpy_filter"].shape[:3]

            # loss_norm = loss / (T * B)
            if self.loss_moving_average is None:
                self.loss_moving_average = loss
            else:
                self.loss_moving_average = \
                    self.ma_param * self.loss_moving_average + (
                            1 - self.ma_param) * loss

            agg_metrics['loss'] = loss
            # agg_metrics['loss_norm'] = loss_norm
            agg_metrics["loss_ma"] = self.loss_moving_average
            if self.store_filter_and_forecast_result:
                agg_metrics["forecast"] = eval_outputs["forecast"]
                # agg_metrics["mpy_filter"] = eval_outputs["mpy_filter"]
                # agg_metrics["Vpy_filter"] = eval_outputs["Vpy_filter"]
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
