import json
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import Forecast


def main():
    """Train a forecast model according to quickstart (https://gluon-ts.mxnet.io/examples/basic_forecasting_tutorial/tutorial.html),
    then compute metrics."""
    dataset, forecasts, tss = quickstart()
    my_evaluator = MyEvaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = my_evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))

    try:
        from IPython.display import display

        with pd.option_context("display.max_rows", 4):
            display(agg_metrics, item_metrics)
    except:  # noqa: E722
        print(json.dumps(agg_metrics, indent=4))
        with pd.option_context("display.max_rows", 4, "display.width", 250):
            print(item_metrics)


class MyEvaluator(Evaluator):
    """This is an example of an evaluator that extends gluonts's one with additional metrics.

    The two metrics here are just toy examples, as they're already part of gluonts, however they're chosen for their
    simplicity and you'd be able to confirm that they produce the same numbers as their original counterpart.

    The key here is to compute two kind of metrics: per-time-series, and the aggregate across all time series. For the
    later part, you just need to think about the proper aggregation function (is it just simply a mean, or another way
    that you want to use).
    """

    @staticmethod
    def mysmape(target, forecast):
        r"""This is SMAPE, but purposely given a new name to demonstrate how to add metric to evaluator.

        .. math::

            smape = mean(2 * |Y - Y_hat| / (|Y| + |Y_hat|))

        https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        """

        denominator = np.abs(target) + np.abs(forecast)
        flag = denominator == 0

        smape = 2 * np.mean((np.abs(target - forecast) * (1 - flag)) / (denominator + flag))
        return smape

    @staticmethod
    def mape(target, forecast):
        r"""
        .. math::
            mape = mean(|Y - Y_hat| / |Y|))
        """
        # https://github.com/awslabs/gluon-ts/pull/725
        denominator = np.abs(target)
        flag = denominator == 0

        mape = np.mean((np.abs(target - forecast) * (1 - flag)) / (denominator + flag))
        return mape

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None]]:
        metrics = super().get_metrics_per_ts(time_series, forecast)

        # region : this is taken from gluonts.evaluation.Evaluator
        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        pred_target = np.ma.masked_invalid(pred_target)
        median_fcst = forecast.quantile(0.5)
        # endregion

        metrics["mysmape"] = self.mysmape(pred_target, median_fcst)
        metrics["mape"] = self.mape(pred_target, median_fcst)  # https://github.com/awslabs/gluon-ts/pull/725
        return metrics

    def get_aggregate_metrics(self, metric_per_ts: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        totals, metric_per_ts = super().get_aggregate_metrics(metric_per_ts)

        # region : this is based on gluonts.evaluator.Evaluator
        agg_funs = {
            "mysmape": "mean",
            "mape": "mean",
        }
        assert set(metric_per_ts.columns) >= agg_funs.keys(), "The some of the requested item metrics are missing."
        my_totals = {key: metric_per_ts[key].agg(agg) for key, agg in agg_funs.items()}
        # endregion

        totals.update(my_totals)
        return totals, metric_per_ts


def quickstart():
    """This is based on the quickstart, but minus the plottings.

    See: https://gluon-ts.mxnet.io/examples/basic_forecasting_tutorial/tutorial.html
    """
    from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
    from gluonts.dataset.util import to_pandas
    from gluonts.evaluation.backtest import make_evaluation_predictions
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.trainer import Trainer

    print(f"Available datasets: {list(dataset_recipes.keys())}")
    dataset = get_dataset("m4_hourly", regenerate=True)

    entry = next(iter(dataset.train))
    train_series = to_pandas(entry)
    entry = next(iter(dataset.test))
    test_series = to_pandas(entry)
    print(f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}")
    print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
    print(f"Frequency of the time series: {dataset.metadata.freq}")

    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=dataset.metadata.prediction_length,
        context_length=100,
        freq=dataset.metadata.freq,
        trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
    )

    predictor = estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    return dataset, forecasts, tss


if __name__ == "__main__":
    main()
