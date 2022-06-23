
# Concepts

## Estimator and Predictor

GluonTS uses two main abstractions called `Estimator` and `Predictor`.

Each `Predictor` implements `.predict(..)`, which given some input time-series
will return forecasts:

```py
forecasts = predictor.predict(data)
```

In contrast, an `Estimator` is trained to produce a `Predictor` which then is
used to make the actual predictions:

```py
predictor = estimator.train(train_data)
forecasts = predictor.predict(data)
```

The reason to split `Estimator` and `Predictor` into two classes is that many
models require a dedicated training step to generate a global model. This
global model is only trained once, but is used to make predictions for all
time-serires.

This is in contrast to local models, which are fitted on individual time-series
and therefore try to capture the characteristics of each time-series but not
the dataset in its entirety.

Training a global model usually takes a lot of time: often hours, but sometimes
even days. Thus, it is not feasable to train the model "offline" (on demand).
In contrast, a local model is usually fitted "oline" as part of the request.

In GluonTS models that are local are available as predictors, whilst global
models are offered as estimators:

```py
# global DeepAR model
estimator = DeepAREstimator(prediction_length=24, freq="H")
predictor = estimator.train(train_data)

# local Prophet model
predictor = ProphetPredictor(prediction_length=24)
```

## Dataset

In GluonTS a `Dataset` is a collection of time-series objects. Each of these
objects has columns (or fields) which represent attributes of the
time-series.

In most models the `target`-field is the column that we want to predict:

```json
{"target": [1, 2, 3, 4, 5, 6]}
```

Note that the `target`-column is not imposed by GluonTS onto models, but it is
used by most models by convention.


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
