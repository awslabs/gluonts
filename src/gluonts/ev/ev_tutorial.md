# New evaluation approach

This document explains how to use the GluonTS `ev` module.

## [Usage](usage_example.py)

### Overview
Before diving into how things work, let's look at how the `ev` module can be used. This can be divided into three steps:
1. Decide on the `test_data` (of type `TestData`) and `predictor` to use.
2. Gather the `metrics` to be evaluated and decide over what `axis` to aggregate (use `None` to aggregate to a single value).
3. Call `predictor.backtest(metrics, test_data, axis)` to get (metric name, metric result) pairs.

### Example

```python
from gluonts.dataset.split import split
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.npts import NPTSPredictor
from gluonts.ev import SumQuantileLoss, MSE
```

First, let's pick a dataset.

```python
dataset = get_dataset("exchange_rate")

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
```

To get the (input, label) pairs of type `TestData`, we do the following:

```python
_, test_template = split(dataset=dataset.test, offset=-prediction_length)
test_data = test_template.generate_instances(prediction_length=prediction_length)
```

Let's use a non-parametric predictor to keep things simple for this example:
```python
predictor = NPTSPredictor(prediction_length=prediction_length, freq=freq)
```

Let's evaluate the Quantile Loss for the 90% quantile as well as the Mean Squared Error. To do this, we call the `backtest` method on our predictor. Choosing `axis=1` evaluates this metric per time series. For more information on how to use `axis`, see [Concepts](#concepts).

```python
evaluation_result = predictor.backtest(
    metrics=[SumQuantileLoss(q=0.9), MSE()],
    test_data=test_data,
    axis=1,
)
```

Finally, we print the `evaluation_result` which is a Python dictionary. This gives for each metric one value per time series.
```python
for name, value in evaluation_result.items():
    print(f"\n{name}: {value}")
```

### Batching
No matter if `Forecast` or `ForecastBatch` objects are passed to `Predictor.backtest`, evaluation happens in batches - `Forecast`s are internally translated into `ForecastBatch`es of batch size 1. This new perspective has the benefit that things work similarly in the univariate and multivariate case. Also, aggregating metrics per time series is possible by just stacking multiple time series with the same time stamps on top of each other.

The interpretation of what such a multidimensional batch means is **not** hardcoded in the `ev` module. Using `Predictor.backtest` however will construct the data to evaluate in a specific way:
- In the univariate case, there are two dimensions and the data is of shape `(num_samples, prediction_length)`.
- In the multivariate case, there is one more dimension for the different variates, meaning the data is of shape `(num_samples, prediction_length, target_dim)`.

## Concepts
### Overview
There are different `Metric`s, like **MSE** or **SumQuantileLoss**. Calling a metric produces an `Evaluator` which can be fed data and eventually asked to give the result.

One characteristic of a metric is that it aggregates the provided data in some way. We distinguish between `Stat`s and `Aggregation`s. 
- A `Stat` is a function that manipulates data in some way that is useful for calculating a metric - it does not aggregate any values.
- An `Aggregation` on the other hand is only good at one thing: aggregating some values over some axis (or all axes).
So, each `Evaluator` makes use of `Stat`s and `Aggregation`s to calculate the desired result.

### [Stats](stats.py)
Stats are simple Python functions that take a `data` dictionary and further parameters as input. One such function represents the heart of one or more metrics and returns an `np.ndarray`. What makes it a "stat" is that there's no aggregation happening here. This is left to `Aggregation`s as explained below.

The provided `data` acts as a dict of `np.ndarrays` with these keys:
- `"label"`: the true time series values to predict
- `"seasonal_error"`: one value per time series representing how similar the input series is to itself shifted by the seasonality (note: even though this aggregates the prediction axis, the number of dimensions is the same as for the other values in `data` - this simplifies calculating the metrics because of NumPy broadcasting)
- `"mean"`: mean forecast
- `"median"`: median forecast (= 50% quantile predictions, `"0.5"`)
- any *`q`* where *`q`* is a string of a number between 0 and 1: a quantile prediction, e.g. for the 95% quantile, use `"0.95"` as *`q`* (note: behind the scenes, dynamic `Forecast` or `ForecastBatch` objects are used so there's no further restriction on how to choose *`q`*)

The idea is that `data` contains all information required to calculate any metric so that everything a metric needs can be read from it.

Note that there are no assumptions made on what the data represents and how many dimensions it has.

### [Time Series Stats](ts_stats.py)
While "stats" take the forecasted values into account somehow, a "time series stat" does not. They act more like attributes of a time series than a comparison between time series and forecast.

Currently, this only includes the seasonal error which is required for metrics like MASE and MSIS.

### [Metrics](metrics.py)
In contrast to stats, metrics always aggregate the underlying data to produce some result. In general, any axis or all of them at once (`axis=None`) can be aggregated - choosing the axis is left to the user.

A `Metric` is defined as a callable which takes as argument the axis to aggregate and returns an `Evaluator` that is able to calculate the desired metric. When a metric has no parameters, it is implemented as a function; otherwise as a class which first needs to be instantiated.

For example, if interested in the quantile loss for the 90% quantile per time series, `quantile_loss = SumQuantileLoss(q=0.9)` is the configured metric and `quantile_loss(axis=1)` produces an `Evaluator` that can be fed the data to evaluate.

### [Evaluators](evaluator.py)
An `Evaluator` is the result of calling a metric. It implements two methods:
- `update` which takes `data` as input and
- `get` which returns the evaluation result of the data provided thus far.

There are two subclasses of `Evaluator` which capture common scenarios of what it means to evaluate a metric.

1. `DirectEvaluator`: This type of evaluator takes two arguments:
   - a `stat` function and
   - an `aggregation` to aggregate the results of calling `stat`.

2. `DerivedEvaluator`: This type of evaluator keeps track of multiple `evaluators` (of type `Dict[str, Evaluator]`) for simpler metrics independently. To calculate the final result, a `post_process` function takes the results of the individual evaluators as input and returns the desired combined result.

### [Aggregations](aggregations.py)
Similar to the evaluators, aggregations have two methods: `update` and `get`. These can be thought of as "map" and "reduce", respectively.

There are two kinds of aggregations provided: `Sum` and `Mean`. To keep track of what data is fed in, `partial_result` takes on one of two roles:
- If axis 0 is aggregated (`axis=0` or `axis=None`), `partial_result` represents the sum so far. For `Sum`, this is directly the result when `get` is called while `Mean` also keeps track of the `n` to divide by in the end.
- Else, `partial_result` is a list that stores the batch-wise aggregations. When `get` is called, `partial_result` is concatenated and returned as a single `np.ndarray`.

## Miscellaneous
### Custom metrics
To create a new metric, first check if the metric can be implemented in such a way that a `DirectEvaluator` or `DerivedEvaluator` is used. If they don't suffice, subclass `Evaluator` directly and overwrite both `update` and `get` to calculate the desired metric.

### Dealing with invalid values
`NaN` and `Inf` values can occur as part of the data or because of invalid operations during metric calculation (like division by zero). By default, these values are **not** treated any different than other values. This is because we don't want to give a false sense of security. Instead, it is the user's responsibility to prepare the data to evaluate on accordingly and choose appropriate metrics for the task.

To skip invalid values, **masking** should be used (for example `np.ma.masked_invalid(original_array)`). Note that once the data is masked, all subsequent calculations will also use masking - this is the normal NumPy behaviour.

So, the `ev` module can work with both `np.ndarray`s as well as `np.ma.masked_array`s. During the mean aggregation, masked values will not be given any weight. For example, the mean of `[1.0, 3.0, --]` will be `2.0`.