"""
This example shows how to fit a model and evaluate its predictions.
"""
import pprint

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer

print(f"datasets available: {dataset_recipes.keys()}")

# we pick m4_hourly as it only contains a few hundred time series
dataset = get_dataset("m4_hourly", regenerate=False)

estimator = SimpleFeedForwardEstimator(
    prediction_length=dataset.metadata.prediction_length,
    freq=dataset.metadata.time_granularity,
    trainer=Trainer(epochs=5, num_batches_per_epoch=10),
)

predictor = estimator.train(dataset.train)

forecast_it, ts_it = make_evaluation_predictions(
    dataset.test, predictor=predictor, num_eval_samples=100
)

agg_metrics, item_metrics = Evaluator()(
    ts_it, forecast_it, num_series=len(dataset.test)
)

pprint.pprint(agg_metrics)
