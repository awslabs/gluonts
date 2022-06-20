# Tutorial on pandas based dataframes dataset

## Introduction

This Tutorial covers how to use gluonts's pandas DataFrame based dataset
`DataFramesDataset`.
We create dummy time series data to illustrate how single and
multiple time series, given in pandas.DataFrame, can be converted
to `gluonts.dataset.dataframe.DataFramesDataset` and used together with
gluonts estimators.


The minimal requirement to start modelling using gluon-ts's dataframe dataset
is a list of monotonically increasing timestamps with a fixed frequency
and a list of corresponding target values in pandas.DataFrame:

|timestamp|target|
|----|----|
|2021-01-01 00:00:00|-0.21|
|2021-01-01 01:00:00|-0.33|
|2021-01-01 02:00:00|-0.33|
|...|...|

If you have additional dynamic or static features, you can include those
in separate columns as following:

|timestamp|target|stat_cat_1|dyn_real_1|
|----|----|----|----|
|2021-01-01 00:00:00|-0.21|0|0.79|
|2021-01-01 01:00:00|-0.33|0|0.59|
|2021-01-01 02:00:00|-0.33|0|0.39|
|...|...|...|...|

Gluonts also supports multiple time series.
Those can either be a list of the DataFrames with the format above
(having at least a `timestamp` index or column and `target` column),
a dict of DataFrames or a Long-DataFrame that has an additional
`item_id` column that groups the different time series.
When using a dict, the keys are used as `item_id`s. An example with
two time series including static and dynamic feature is the following:

|timestamp|target|item_id|stat_cat_1|dyn_real_1|
|----|----|----|----|----|
|2021-01-01 00:00:00|-0.21|A|0|0.79|
|2021-01-01 01:00:00|-0.33|A|0|0.59|
|2021-01-01 02:00:00|-0.33|A|0|0.39|
|2021-01-01 01:00:00|-1.24|B|1|-0.60|
|2021-01-01 02:00:00|-1.37|B|1|-0.91|

Here, the time series values are represented by the `target` column with
corresponding timestamp values in the `timestamp` column. Another necessary column
is the `item_id` column that indicates which time series a specific row belongs to.
In this example, we have two time series, one indicated by item `A` and the other
by `B`. In addition, if we have other features we can include those too
(`stat_cat_1` and `dyn_real_1` in the example above).

## Dummy data generation

Now, let's create a function that generates dummy time series conforming above
stated requirements (`timestamp` index and `target` column).
The function randomly samples sin/cos curves and outputs their sum. You don't need
to understand how this really works. We will just call `generate_single_ts`
with datetime values which rise monotonically with fixed frequency and get
a pandas.DataFrame with `timestamp` and `target` values.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_single_ts(date_range, item_id=None) -> pd.DataFrame:
    """create sum of `n_f` sin/cos curves with random scale and phase."""
    n_f = 2
    period = np.array([24/(i+1) for i in range(n_f)]).reshape(1, n_f)
    scale = np.random.normal(1, 0.3, size=(1, n_f))
    phase = 2*np.pi*np.random.uniform(size=(1, n_f))
    periodic_f = lambda x: scale*np.sin(np.pi*x/period + phase)

    t = np.arange(0, len(date_range)).reshape(-1,1)
    target = periodic_f(t).sum(axis=1) + np.random.normal(0, 0.1, size=len(t))
    ts = pd.DataFrame({"target": target}, index=date_range)
    if item_id is not None: ts["item_id"] = item_id
    return ts
```


```python
prediction_length, freq = 24, "1H"
T = 10*prediction_length
date_range = pd.date_range("2021-01-01", periods=T, freq=freq)
ts = generate_single_ts(date_range)

print("ts.shape:", ts.shape)
print(ts.head())
ts.loc[:, "target"].plot(figsize=(10, 5))
```

## Use single time series for training

Now, we create a gluonts dataset using the time series, we generated and train
a model. Because we will use the train/evaluation loop multiple times, let's
create a function for it. The input to the function will be the time series
in multiple formats and the put is the `MSE` (mean squared error). In this
function we create a model, train it to get the predictor, use the predictor
to create forecasts and return a metric. Here, we run the training for just
1 epoch. Usually you need more epochs to fully train the model.




```python
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator

def train_and_predict(dataset, prediction_length):
    model = DeepAREstimator(
        freq=dataset.freq,
        prediction_length=prediction_length,
        trainer=Trainer(epochs=1),
    )
    predictor = model.train(dataset)
    forecast_it, ts_it = make_evaluation_predictions(dataset=ds, predictor=predictor)
    evaluator = Evaluator(quantiles=(np.arange(20) / 20.0)[1:])
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(dataset))
    return agg_metrics["MSE"]
```

We are ready to convert the `ts` dataframe into gluonts dataset and train
a model. For that, we import the `DataFramesDataset` and create an instance
using our time series `ts`. If the `target`-column is called "target"
we don't really need to provide it to the constructor. Also `freq` is
inferred from the data if the timestamp is a time- or period range.
However, in this tutorial we will be specific on 
how to use those in general.


```python
from gluonts.dataset.dataframe import DataFramesDataset

ds = DataFramesDataset(ts, target="target", freq=freq)
train_and_predict(ds, prediction_length)
```

## Use multiple time series for training

As described above, we can also use a list of single time series
dataframes. Even dict of dataframes or a single long-formatted-dataframe
with multiple time series can be used and is described below.
So, lets create multiple time series and train the model
using those. 


```python
N = 10
multiple_ts = [generate_single_ts(date_range) for i in range(N)]

ds = DataFramesDataset(multiple_ts, target="target", freq=freq)
train_and_predict(ds, prediction_length)
```

If the dataset is given as a long-dataframe, we can also use an
alternative constructor `from_long_dataframe`. Note in this case
we have to provide the `item_id` as well. Also note, if the
`item_id`-column is called "item_id", we don't need to explicitly
pass it to the constructor.


```python
N = 10
ts_in_long_format = pd.concat(
    [generate_single_ts(date_range, item_id=i) for i in range(N)]
)

# Note we need an item_id column now and provide its name (default is "item_id") to
# the constructor. Otherwise, there is no way to distinguish different time series.
ds = DataFramesDataset.from_long_dataframe(
    ts_in_long_format, item_id="item_id", target="target", freq=freq
)
train_and_predict(ds, prediction_length)
```

## Include static and dynamic features

The DataFramesDataset also allows us to include features of our time series.
The list of all available features is given in the documentation.
Here, we use dynamic real and static categorical features.
To mimic those features we will just wrap our fancy data generator
`generate_single_ts` and add a static cat and two dynamic real features.


```python
def generate_single_ts_with_features(date_range, item_id) -> pd.DataFrame:
    ts = generate_single_ts(date_range, item_id)
    T = ts.shape[0]
    # static features are constant for each series
    ts["static_cat_1"] = np.random.randint(low=0, high=2) # high is exclusive
    ts["dynamic_real_1"] = np.random.normal(size=T)
    ts["dynamic_real_2"] = np.random.normal(size=T)
    # ... we can have as many static or dynamic features as we like
    return ts

ts = generate_single_ts_with_features(date_range, item_id=0)
ts.head()
```

Now, when we create the gluonts dataset, we need to let the constructor
know which columns are the categorical and real features. Also, we need a
minor modification to the `train_and_predict` function. Here, we need to
let the model know as well that more features are coming in. Note, if
categorical features are provided, DeepAR also needs the cardinality as input.
We have one static categorical feature that can take on two
values (0 or 1). The cardinality in this case is a list of one element `[2,]`.


```python
def train_and_predict_with_features(ds, prediction_length):
    model = DeepAREstimator(
        freq=ds.freq,
        prediction_length=prediction_length,
        use_feat_dynamic_real=True,
        use_feat_static_cat=True,
        cardinality=[2,],
        trainer=Trainer(epochs=1),
    )
    predictor = model.train(ds)
    forecast_it, ts_it = make_evaluation_predictions(dataset=ds, predictor=predictor)
    evaluator = Evaluator(quantiles=(np.arange(20) / 20.0)[1:])
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(ds))
    return agg_metrics["MSE"]
```

Lets now generate single time series and multiple time series
(as list and as long-dataframe).
Note we don't provide `freq` and `timestamp`, because they are automatically
inferred from the index. We are also not passing "target" and "item_id"
(for long-dataframe) because those are the default values of `target`
and `item_id`, respectively.


```python
single_ts = generate_single_ts_with_features(date_range, item_id=0)
multiple_ts = [generate_single_ts_with_features(date_range, item_id=i) for i in range(N)]
multiple_ts_long = pd.concat(multiple_ts)

single_ts_dataset = DataFramesDataset(
    single_ts,
    feat_dynamic_real=["dynamic_real_1", "dynamic_real_2"],
    feat_static_cat=["static_cat_1"],
)
multiple_ts_dataset = DataFramesDataset(
    multiple_ts,
    feat_dynamic_real=["dynamic_real_1", "dynamic_real_2"],
    feat_static_cat=["static_cat_1"],
)
multiple_ts_dataset_dict = DataFramesDataset(
    {i: _ts for i, _ts in enumerate(multiple_ts)},
    feat_dynamic_real=["dynamic_real_1", "dynamic_real_2"],
    feat_static_cat=["static_cat_1"],
)
# for long-dataset we use a different constructor and need a `item_id` column
multiple_ts_long_dataset = DataFramesDataset.from_long_dataframe(
    multiple_ts_long,
    feat_dynamic_real=["dynamic_real_1", "dynamic_real_2"],
    feat_static_cat=["static_cat_1"],
)
```

Thats it! We can now call `train_and_predict_with_features` with all the datasets.


```python
train_and_predict_with_features(single_ts_dataset, prediction_length)
```

```python
train_and_predict_with_features(multiple_ts_dataset, prediction_length)
```

```python
train_and_predict_with_features(multiple_ts_dataset_dict, prediction_length)
```

```python
train_and_predict_with_features(multiple_ts_long_dataset, prediction_length)
```

## Use train/test split

Here, we split the DataFrame/DataFrames into training and test data
using the `OffsetSplitter` from gluonts. We can then
use the training data to train the model and the test data for prediction.
Note, the `OffsetSplitter` needs the `item_id`. If you want to split
by date you can also use the `DateSplitter`.


```python
prediction_length, freq = 24, "1H"
T = 10*prediction_length
date_range = pd.date_range("2021-01-01", periods=T, freq=freq)
ts = generate_single_ts(date_range, item_id=0)
```


```python
from gluonts.dataset.split import OffsetSplitter # DateSplitter

#Note the minus-sign for `split_offset`. This will split the time series counting from the end.
splitter = OffsetSplitter(prediction_length=prediction_length, split_offset=-prediction_length)
split = splitter.split(DataFramesDataset(ts, item_id="item_id"))
train, test = split.train, split.test
```


```python
model = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    trainer=Trainer(epochs=1),
)
predictor = model.train(train)
forecast_it, ts_it = make_evaluation_predictions(dataset=test, predictor=predictor)
evaluator = Evaluator(quantiles=(np.arange(20) / 20.0)[1:])
agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(test))
```

## Train and visualize forecasts

Lets generate again some dummy data with only static categorical features,
train an estimator and visualize the forecasts.


```python
prediction_length, freq = 24, "1H"
T = 10*prediction_length
date_range = pd.date_range("2021-01-01", periods=T, freq=freq)

N = 100
time_seriess = [generate_single_ts(date_range, item_id=i) for i in range(N)]
dataset = DataFramesDataset(time_seriess, freq=freq, item_id="item_id")

splitter = OffsetSplitter(prediction_length=prediction_length, split_offset=-prediction_length)
split = splitter.split(dataset)
train, test = split.train, split.test
```


```python
model = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    trainer=Trainer(epochs=10),
)
predictor = model.train(train)
forecast_it, ts_it = make_evaluation_predictions(dataset=test, predictor=predictor)
forecasts = list(forecast_it)
tests = list(ts_it)
evaluator = Evaluator(quantiles=(np.arange(20) / 20.0)[1:])
agg_metrics, item_metrics = evaluator(tests, forecasts, num_series=len(test))
```

Let's plot a few randomly chosen series.

```python
n_plot = 3
indices = np.random.choice(np.arange(0, N), size=n_plot, replace=False)
fig, axes = plt.subplots(n_plot, 1, figsize=(10, n_plot*5))
for index, ax in zip(indices, axes):
    tests[index][-4*prediction_length:].plot(ax=ax)
    plt.sca(ax)
    forecasts[index].plot(prediction_intervals=[90.], color='g')
    plt.legend(["observed", "predicted median", "predicted 90% quantile"])
```