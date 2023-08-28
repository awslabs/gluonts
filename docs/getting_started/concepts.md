
# Concepts

## Estimator and Predictor

GluonTS uses two main abstractions called `Estimator` and `Predictor`.

Each `Predictor` implements `.predict(..)` which, given some input time series,
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
time series.

This is in contrast to local models, which are fitted on individual time series
and therefore try to capture the characteristics of each time series but not
the dataset in its entirety.

Training a global model can take a lot of time: up to hours, but sometimes even
days. Thus, it is not feasible to train the model as part of the prediction
request and it happens as a separate "offline" step. In contrast, fitting a
local model is usually much faster and is done "online" as part of the
prediction.

In GluonTS, local models are directly available as predictors, whilst global
models are offered as estimators, which need to be trained first:

```py
# global DeepAR model
estimator = DeepAREstimator(prediction_length=24, freq="H")
predictor = estimator.train(train_data)

# local Prophet model
predictor = ProphetPredictor(prediction_length=24)
```

## Dataset

In GluonTS, a `Dataset` is a collection of time series objects. Each of these
objects has columns (or fields) which represent attributes of the
time series.

Most models use the `target` column to indicate the time series that we want to
predict in the future:

```json
{"target": [1, 2, 3, 4, 5, 6]}
```

Note that the `target` column is not imposed by GluonTS onto models, but it is
used by most models by convention.


### API

To be more precise, a `Dataset` is defined like this:

```py
DataEntry = dict[str, Any]

class Dataset(Protocol):
    def __iter__(self) -> Iterator[DataEntry]:
        ...

    def __len__(self) -> int:
        ...
```

In other words, anything that can emit dictionaries can act as a `Dataset`.
