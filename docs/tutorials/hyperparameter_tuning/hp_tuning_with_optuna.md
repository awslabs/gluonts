# Tuning GluonTS models with Optuna

In this notebook we will see how to tune the hyperparameters of a GlutonTS model using Optuna. For this example, we are going to tune a PyTorch-based DeepAREstimator.

## Data loading and processing


```python
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
```

### Provided datasets


```python
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
```


```python
print(f"Available datasets: {list(dataset_recipes.keys())}")
```


```python
dataset = get_dataset("electricity", regenerate=True)
```

### Extract and split training and test data sets

In general, the datasets provided by GluonTS are objects that consists of three main members:

- `dataset.train` is an iterable collection of data entries used for training. Each entry corresponds to one time series
- `dataset.test` is an iterable collection of data entries used for inference. The test dataset is an extended version of the train dataset that contains a window in the end of each time series that was not seen during training. This window has length equal to the recommended prediction length.
- `dataset.metadata` contains metadata of the dataset such as the frequency of the time series, a recommended prediction horizon, associated features, etc.

To keep the example small and quick to execute, we are only going to use a subset of the dataset.


```python
from itertools import islice
electricity_train_sub = list(islice(dataset.train, 10))
electricity_test_sub = list(islice(dataset.test, 15))
```


```python
for train_series in electricity_train_sub:
    train_series['target'] = train_series['target'][:2400]
for test_series in electricity_test_sub:
    test_series['target'] = test_series['target'][:2424]
```


```python
print(electricity_train_sub)
```

Check out the details of the `dataset.metadata`.


```python
print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
print(f"Frequency of the time series: {dataset.metadata.freq}")
```

Check and plot the result of electricity data subset 


```python
train_series = to_pandas(electricity_train_sub[0])
train_series.plot()
plt.grid(which="both")
plt.legend(["train series"], loc="upper left")
plt.show()
```


```python
test_series = to_pandas(electricity_test_sub[0])
test_series.plot()
plt.axvline(train_series.index[-1], color='r') # end of train dataset
plt.grid(which="both")
plt.legend(["test series", "end of train series"], loc="upper left")
plt.show()
```


```python
print(f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}")
```

## Tuning parameters of DeepAR estimator


```python
import optuna
import torch
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import Evaluator
```

An example that tune the DeepAR estimator on electricity dataset via Optuna. We choose two hyperparameters `num_layers` and `hidden_size` to optimize.

Define an Objective class used in tuning process of Optuna.
With `__init__` function, initialize the dataset, prediction_length, freq and metric_type(means what metric to be used to evaluate the DeepAREstimator).
With `get_params` function, define what hyperparameters to be tuned within given range.
With `split_entry` function, split each time series of the dataset into two part:
- entry_past: the training part 
- entry_future: the label part used in validation
With `__call__` function, define the way DeepAREstimator is used in training and validation.


```python
class DeepARTuningObjective:  
    def __init__(self, dataset, prediction_length, freq, metric_type="mean_wQuantileLoss"):
        self.dataset = dataset
        self.prediction_length = prediction_length
        self.freq = freq
        self.metric_type = metric_type
    
    def get_params(self, trial) -> dict:
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 5),
            "hidden_size": trial.suggest_int("hidden_size", 10, 50),
        }

    def split_entry(self, entry):
        entry_past = {}
        for key, value in entry.items():
            if key == "target":
                entry_past[key] = value[: len(value) - self.prediction_length]
            else:
                entry_past[key] = value

        df = pd.DataFrame(entry['target'], columns=[entry['item_id']])
        df = df.set_index(pd.date_range(start=entry['start'], 
                                        periods=len(entry['target']), freq=self.freq))
        return entry_past, df[-self.prediction_length:]
     
    def __call__(self, trial):
        params = self.get_params(trial)
        estimator = DeepAREstimator(
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size'],
            prediction_length=self.prediction_length, 
            freq=self.freq,
            trainer_kwargs={
                "progress_bar_refresh_rate": 0, 
                "weights_summary": None, 
                "max_epochs": 5, 
            }
        )
        
        entry_splited = [self.split_entry(entry) for entry in self.dataset]
        entry_pasts = [entry[0] for entry in entry_splited]
        entry_futures = [entry[1] for entry in entry_splited]
        
        predictor = estimator.train(entry_pasts, cache_data=True)
        forecast_it = predictor.predict(entry_pasts)
        
        forecasts = list(forecast_it)
        tss = list(entry_futures)
        
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(self.dataset))
        return agg_metrics[self.metric_type]
```

Implement the Optuna tuning process.


```python
import time
start_time = time.time()
study = optuna.create_study(direction="minimize")
study.optimize(Objective(electricity_train_sub, dataset.metadata.prediction_length, dataset.metadata.freq),
               n_trials=10)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
print(time.time() - start_time)
```

# Training deepar estimator

After getting the best hyperparameters by optuna,you can set them into the DeepAR estimator to realize the prediction of electricity dataset.

## Training and predict

Now We can begin with GulonTS's pre-built DeepAR estimator after tuning its hyperparameters by optuna. The next process consists of training a model, producing forecasts, and evaluating the results.


```python
from gluonts.mx import Trainer
```


```python
estimator = DeepAREstimator(
    num_layers=trial.params["num_layers"],
    hidden_size=trial.params["hidden_size"],
    prediction_length=dataset.metadata.prediction_length,
    context_length=100,
    freq=dataset.metadata.freq,
    trainer_kwargs={
        "progress_bar_refresh_rate": 0, 
        "weights_summary": None, 
        "max_epochs": 5,
    }
)
```

After specifying our estimator with all the necessary hyperparameters we can train it using our training dataset `electricity_train` by invoking the `train` method of the estimator. The training algorithm returns a fitted model (or a `Predictor` in GluonTS parlance) that can be used to construct forecasts.


```python
predictor = estimator.train(electricity_train_sub, cache_data=True)
```

## Visualize and evaluate forecasts

With a predictor in hand, we can now predict the last window of the `electricity.test` and evaluate our model's performance.

GluonTS comes with the `make_evaluation_predictions` function that automates the process of prediction and model evaluation. Roughly, this function performs the following steps:

- Removes the final window of length `prediction_length` of the `electricity .test` that we want to predict
- The estimator uses the remaining data to predict (in the form of sample paths) the "future" window that was just removed
- The module outputs the forecast sample paths and the `electricity .test` (as python generator objects)


```python
from gluonts.evaluation import make_evaluation_predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=electricity_test_sub,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
```

First, we can convert these generators to lists to ease the subsequent computations.


```python
forecasts = list(forecast_it)
tss = list(ts_it)
```

We can examine the first element of these lists (that corresponds to the first time series of the dataset). Let's start with the list containing the time series, i.e., `tss`. We expect the first entry of `tss` to contain the (target of the) first time series of `electricity.test`.


```python
# first entry of the time series list
ts_entry = tss[0]
# first entry of the forecast list
forecast_entry = forecasts[0]
```

`Forecast` objects have a `plot` method that can summarize the forecast paths as the mean, prediction intervals, etc. The prediction intervals are shaded in different colors as a "fan chart".


```python
def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()
```


```python
plot_prob_forecasts(ts_entry, forecast_entry)
```

We can also evaluate the quality of our forecasts numerically. In GluonTS, the `Evaluator` class can compute aggregate performance metrics, as well as metrics per time series (which can be useful for analyzing performance across heterogeneous time series).


```python
from gluonts.evaluation import Evaluator
```


```python
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(electricity_test_sub))
```

Aggregate metrics aggregate both across time-steps and across time series.


```python
print(json.dumps(agg_metrics, indent=4))
```

Individual metrics are aggregated only across time-steps.


```python
item_metrics.head()
```


```python
item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
plt.grid(which="both")
plt.show()
```
