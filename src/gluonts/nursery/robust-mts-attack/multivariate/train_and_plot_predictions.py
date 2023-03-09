# Standard library imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gluonts.evaluation import MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.multivariate.datasets.dataset import (
    exchange_rate,
    random_periodic,
    solar, electricity)
from gluonts.multivariate.hyperparams import Hyperparams
from gluonts.multivariate.multivariate_models import models_dict


def plot(target, forecast, prediction_length):
    rows = 8
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(6, 12))
    axx = axs.ravel()
    seq_len, target_dim = target.shape
    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        target[-3 * prediction_length :][dim].plot(ax=ax)

        # (quantile, target_dim, seq_len)
        pred_df = pd.DataFrame(
            {q: forecast.quantile(q)[dim] for q in [0.1, 0.5, 0.9]},
            index=forecast.index,
        )

        ax.fill_between(
            forecast.index, pred_df[0.1], pred_df[0.9], alpha=0.2, color='g'
        )
        pred_df[0.5].plot(ax=ax, color='g')
    plt.show()


def plot_predictions(dataset, predictor):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test_ds, predictor=predictor, num_eval_samples=100
    )

    print("predicting")
    forecasts = list(forecast_it)
    targets = list(ts_it)

    plot(
        target=targets[0],
        forecast=forecasts[0],
        prediction_length=dataset.prediction_length,
    )
    plt.show()

    # evaluate
    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
    )

    agg_metrics, item_metrics = evaluator(
        targets, forecasts, num_series=len(dataset.test_ds)
    )

    print(agg_metrics)
    print("CRPS:", agg_metrics["mean_wQuantileLoss"])
    print("CRPS-sum:", agg_metrics["m_sum_mean_wQuantileLoss"])


if __name__ == '__main__':
    ds = electricity()
    params = Hyperparams(hybridize=True)

    estimator = models_dict["GPCOP"](
         freq=ds.freq,
         prediction_length=ds.prediction_length,
         target_dim=ds.target_dim,
         params=params,
    )

    predictor = estimator.train(ds.train_ds)
    plot_predictions(
         dataset=ds,
         predictor=predictor,
    )
