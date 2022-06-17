
# Concepts

## Estimator and Predictor

In GluonTS we use two main abstractions called `Estimator` and `Predictor`.

Each `Predictor` implements a `predict` method, which given some input time-series will return forecasts:

```py
forecasts = predictor.predict(data)
```

In contrast, an `Estimator` can be trained to produce a `Predictor` which then is used to make predictions:

```py
predictor = estimator.train(train_data)
forecasts = predictor.predict(data)
```

The reason to split `Estimator` and `Predictor` into two classes is that many models use a dedicated training step to generate a global model.

While a local model is used to make predictions for a single time-series, there is only one global model to make predictions for all time-series for a given scenario.

Training a global model usually takes a lot of time (up to days), but it is very fast to make predictions given an already trained model. In contrast, a local model is usually fitted on each invocation and while it doesn't require a dedicated training step, they are often slower when making predictions.

In GluonTS models that are local are available as predictors, whilst global models are offered as estimators:

```py
estimator = DeepAREstimator(prediction_length=24, freq="H")
predictor = estimator.train(train_data)

predictor = ProphetPredictor(prediction_length=24)
```


## Dataset

In GluonTS a `Dataset` is a collection of time-series objects. Each of these objects has columns (or fields) which represent attributes of the time-series.

In most models the `target`-field is the column that we want to predict:

```json
{"target": [1, 2, 3, 4, 5, 6]}
```

Note that the `target`-column is not imposed by GluonTS onto models, but it is used by most models by convention.


### API

To be more precise, a `Dataset` is defined as:

```py
DataEntry = dict[str, Any]

class Dataset(Protocol):
    def __iter__(self) -> Iterator[DataEntry]:
        ...

    def __len__(self) -> int:
        raise ...
```

In other words, anything that can emit dictionaries can act as a `Dataset`.
