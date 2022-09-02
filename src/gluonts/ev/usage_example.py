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


evaluator = Evaluator(
    [
        MSE(),
        MSE(aggr="mean"),
        MSE(aggr="sum"),
        Mape(aggr="sum"),
        Mape(),
        MSE(aggr=sum_of_last_ten),
    ]
)

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

100%|██████████| 100/100 [00:00<00:00, 180.28it/s, epoch=1/1, avg_epoch_loss=6.14]
     item_id            mse      mape
0          0    6332.337402  0.106487
1          1  578881.000000  0.314353
2          2   17221.837891  0.082568
3          3  160789.437500  0.068083
4          4   75936.507812  0.103300
..       ...            ...       ...
409      409    2292.286865  0.699178
410      410    3170.718506  1.031616
411      411    2601.059570  0.951782
412      412     111.052147  0.260463
413      413     372.103760  0.415241

[414 rows x 3 columns]
{'mse_mean': 16916181.978052687, 'mse_sum': 7003299338.913813, 'mape_sum': 179.80691988021135, 'mse_sum_of_last_ten': 174118.18099212646}
"""
