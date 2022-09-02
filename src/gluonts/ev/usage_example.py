from typing import Dict

import numpy as np
import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MSE, Mape

from gluonts.ev.evaluator import Evaluator
from gluonts.mx import SimpleFeedForwardEstimator, Trainer

# SETUP

dataset = get_dataset(
    "m4_hourly"
).train  # let's pretend, this is the entire dataset
prediction_length = get_dataset("m4_hourly").metadata.prediction_length

training_dataset, test_template = split(
    dataset, date=pd.Period("1750-01-07 00:00:00", freq="1H")
)

test_pairs = test_template.generate_instances(
    prediction_length=prediction_length
)

estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=prediction_length,
    trainer=Trainer(
        ctx="cpu", epochs=1, learning_rate=1e-3, num_batches_per_epoch=100
    ),
)

predictor = estimator.train(training_dataset)
forecast_it = predictor.predict(
    dataset=test_pairs.input  # expected datatype is Dataset so this is not the best way to do it...
)


# EVALUATION
# let's get the MSE per entry as well as aggregated and also the mean MAPE
# TODO: this might not work because metrics aren't properly sorted yet in Evaluator

# define a custom aggregation function
def sum_of_last_ten(values: np.ndarray) -> float:
    return np.sum(values[-10:]).item()


evaluator = Evaluator([MSE(aggr="mean")])

local_metrics = evaluator.apply(
    test_pairs, forecast_it
)  # returns a LocalMetrics object
print(pd.DataFrame(local_metrics.get()).rename_axis("item_id").reset_index())

global_metrics = (
    local_metrics.aggregate()
)  # todo: returns a dict, should maybe also return a Metrics object
print(global_metrics)

"""
RESULT:

100%|██████████| 100/100 [00:00<00:00, 177.19it/s, epoch=1/1, avg_epoch_loss=6.25]
Empty DataFrame
Columns: [item_id]
Index: []
{'mse_mean': 12172456.495098885}
"""
