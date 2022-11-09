# New evaluation approach

This notebook explains how to use the `ev` module which is a new alternative to the `evaluation` module in GluonTS.

## [Usage](usage_example.py)

Before diving into how things work, let's look at how to use the new evaluation approach.

### Overview
1. Decide on the `test_data` (of type `TestData`) and `predictor` to use.
2. Gather the `metrics` to be evaluated and decide over what `axis` to aggregate (use `None` to aggregate to a single value).
3. Call `predictor.backtest(metrics, test_data, axis)` to get (metric name, metric result) pairs.

### Example

```
from gluonts.dataset.split import split
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.npts import NPTSPredictor
from gluonts.ev.metrics import MSIS, MSE, SumQuantileLoss
```

First, let's pick a dataset and predictor:
```
dataset = get_dataset("exchange_rate")

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq

_, test_template = split(dataset=dataset.test, offset=-prediction_length)
test_data = test_template.generate_instances(
    prediction_length=prediction_length
)

predictor = NPTSPredictor(prediction_length=prediction_length, freq=freq)
```

Next, we specify what metrics to evaluate. We don't provide arguments to MSE and MSIS because we are fine with the default values (in this case, taking the mean forecast as reference for MSE and using alpha=0.05 for MSIS).
```
metrics_per_entry = [MSE(), SumQuantileLoss(q=0.9), MSIS()]
```
To evaluate these metrics, we call the `backtest` method on our predictor. We want to have these metrics evaluated per time series so we specify `axis=1 `(an `axis` of `0` would aggregate per timestamp and `None` would aggregate overall). The `num_samples` is an optional argument used during the call to `predict`.
```
evaluation_result = predictor.backtest(
    metrics=metrics_per_entry, test_data=test_data, axis=1, num_samples=100
)
```
Finally, we print the `evaluation_result`.
```
for name, value in evaluation_result.items():
    print(f"\n{name}: {value}")
```
This gives:
```
MSE: [2.42091992e-03 2.96873403e-04 2.32838181e-04 4.56364661e-04
 4.88960640e-05 1.81302814e-07 2.58689462e-03 2.14975906e-03
 2.92379305e-03 7.49952987e-04 3.14545053e-04 7.56220293e-04
 5.10044955e-05 1.52030675e-08 3.26244101e-03 2.08630866e-03
 2.82381139e-03 6.12264155e-04 3.14068729e-04 1.03861000e-03
 4.65201312e-05 6.37289495e-07 3.23557078e-03 1.48686040e-03
 1.23413809e-03 7.47005498e-04 5.97453674e-05 1.13402517e-03
 3.91843199e-05 1.79914710e-06 3.95487177e-03 8.02376727e-04
 8.99772338e-04 5.70332479e-03 4.58576265e-04 1.38216024e-04
 3.69498352e-05 2.30762632e-06 1.89816125e-03 4.27966194e-04]

sum_quantile_loss[0.9]: [0.16866145 0.22557795 0.14587718 0.5725431  0.08625072 0.002509
 0.07869875 0.22347358 0.10413065 0.14404998 0.10924783 0.52315023
 0.09618294 0.004901   0.35419931 0.20256658 0.09526651 0.16359525
 0.09473243 0.40709562 0.08584013 0.0090656  0.39686875 0.02707392
 0.15419419 0.39663935 0.18439691 0.36972213 0.03120218 0.012554
 0.53890196 0.05934963 0.18462441 0.72134459 0.30531565 0.49801474
 0.03203619 0.0137856  0.05998336 0.09169762]

MSIS: [ 28.47636274  10.68124685  19.50417015  36.81182074 280.62059249
  23.41230718  22.66611657  38.78746667  26.45110228  11.43631121
  19.01874257  37.59945338 199.62173272  22.74449539  29.08663882
  37.83287725  25.41633501  11.03428621  17.14530506  36.30222833
 109.94536901  22.25100276  23.71050878  35.08409363  24.394528
  10.6051624   16.58757618  36.13923834  79.2351969   40.7734285
  29.80105103  33.59646885  24.89069135  13.80472303  16.12984192
  37.42225089 100.73339766  84.97893271  21.19676102  34.13306657]
```

### Batching
No matter if `Forecast` or `ForecastBatch` objects are passed to `Predictor.backtest`, evaluation happens in batches - `Forecast`s are internally translated into `BatchForecast`s of batch size 1. This new perspective has the benefit that things work very similar in the univariate and multivariate case and there's no need for heavily differentiating between `Evaluator` and `MultivariateEvaluator` as it's done in the original approach. Also, aggregating metrics per time series becomes possible by just stacking multiple time series with the same time stamps on top of each other.

The interpretation of what such a multidimensional batch means is **not** hardcoded in the `ev` module. Using `Predictor.backtest` however will construct the data to evaluate in a specific way:
- In the univariate case, there are two dimensions and the data is of shape (num_samples, prediction_length).
- In the multivariate case, there is one more dimension for the different variates, meaning data is of shape (num samples, prediction length, target dim). ([WIP](https://github.com/awslabs/gluonts/pull/2352))

## Concepts
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

## Metric support
### Custom metrics
To create a new metric, first check if the metric can be implemented in such a way that a `DirectEvaluator` or `DerivedEvaluator` is used. If they don't suffice, subclass `Evaluator` directly and overwrite both `update` and `get` to calculate the desired metric.

## Open To Dos
### Handling invalid values
While calculating the metrics, `NaN` values are currently ignored in the aggregations. Two things should be improved here:
- One should be able to choose if values should be ignored or not.
- It should be possible to also ignore `Inf` values.

For comparison, the old evaluation approach has an option to `ignore_invalid_values` (in which case, input values are masked and those masks propagate through all subsequent calculations) as well as three modes of aggregation: `aggregate_all`, `aggregate_no_nan` and `aggregate_valid`.

So, the question is how to transparently deal with invalid values.
