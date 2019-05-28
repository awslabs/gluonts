"""
This example shows how to fit a model and evaluate its predictions.
"""
import pprint
from functools import partial

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.ar2n2 import AR2N2Estimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
import pandas as pd

datasets = [
    "m4_hourly",
    "m4_daily",
    "m4_weekly",
    "m4_monthly",
    "m4_quarterly",
    "m4_yearly",
]

epochs = 100
num_batches_per_epoch = 50

estimators = [
    partial(SimpleFeedForwardEstimator, trainer=Trainer(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch)),
    #partial(AR2N2Estimator, trainer=Trainer(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch)),
    #partial(MQCNNEstimator, trainer=Trainer(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch)),
]

results = []
for dataset_name in datasets:
    for estimator in estimators:

        dataset = get_dataset(dataset_name)

        estimator = estimator(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.time_granularity,
        )

        print(f"evaluating {estimator} on {dataset}")

        predictor = estimator.train(dataset.train)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset.test, predictor=predictor, num_eval_samples=100
        )

        agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(dataset.test))

        pprint.pprint(agg_metrics)

        res = agg_metrics
        res["dataset"] = dataset_name
        res["estimator"] = type(estimator).__name__

        results.append(res)

df = pd.DataFrame(results)

sub_df = df[["dataset", "estimator", "RMSE", "mean_wQuantileLoss", "MASE", "sMAPE", "MSIS"]]

print(sub_df.to_string())