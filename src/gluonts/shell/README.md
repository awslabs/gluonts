
# gluonts.shell

The `shell` module integrates gluon-ts with Amazon SageMaker.

It's design is based on [SageMaker DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html).

This means that interacting with containers built with the shell-module behave
similar to the official DeepAR image.


## Building a container

Building a container is simple. GluonTS and model-specific dependencies need to
be installed.

A minimal Dockerfile looks like this:

```Dockerfile
FROM python:3.7

# "[shell]" ensures that shell-dependencies are installed
RUN pip install gluonts[shell]

# This line is crucial. We need to set `gluonts.shell` as entry-point to
# dispatch the `train` and `serve` commands.
ENTRYPOINT ["python", "-m", "gluonts.shell"]
```


## Forecaster

In GluonTS we distinguish between `Estimator`s and `Predictor`s. An `Estimator`
cannot be used to make predictions directly, but needs to be trained first. The
output of that training is an `Predictor`.

However, there are models which not need to be trained and are therefore
directly represented by `Predictors`. For example, the `model.trivial.consant`
module contains a `ConstantValuePredictor` which just outputs a static value as
prediction.

It can still be useful to run training jobs on `Predictors` for metric
calculation.

We therefore use the term `Forecaster` to mean either `Estimator` or
`Predictor`:

```python
Forecaster = Type[Union[Estimator, Predictor]]
```

## Training Jobs

### Specifying the forecater

A container built with the shell-module is not automatically bind to a
`Forecaster`. There are two ways to do this:

Define the forecaster-class in the Dockerfile:

    ENV GLUONTS_FORECASTER=gluonts.model.deepar.DeepAREstimator


Pass the forecaster-class as a hyper-parameter:

```python
{
    ...
    "forecaster_name": "gluonts.model.deepar.DeepAREstimator",
    ...

}
```

**Note:** The entire class-path needs to be passed.

For more details, see `train.py`.

### Hyper-parameters

All hyper-parameters are passed to the Forecaster directly.

Arguments to a trainer, should be passed in flat:

```python
{
    ...
    # estimator-argument
    "prediction_length": ...,
    # trainer-argument
    "epochs": 100,
    ...
}
```


## Inference

### Trained Models

Models, which were trained using a training-job can simply be used to create an
endpoint.

Inference works similarly to
[*DeepAR* in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar-in-formats.html).

We use the same json-format of `instances` and `configuration`.

See `serve/app.py` for details.

### Using Predictors

It is also possible to run predictors directly in inference. Here, it is
important to pass all parameters to the `Predictor` class as part of the
`configuration`:

```python
{
    "instances": [...],
    "configuration": {
        "forecaster_name": "gluonts.model.trivial.constant.ConstantValuePredictor",
        "prediction_length": 7,
        "freq": "D",
        "value": 0.0,
        ...
    }
}
```

## Batch Transform Jobs

Batch transform works similar to how it does in [SageMaker DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar-in-formats.html#deepar-batch).

The input is expected to be in jsonlines format.

The main difference to SageMaker DeepAR is that GluonTS uses `INFERENCE_CONFIG`
as the environment-variable, instead of `DEEPAR_INFERENCE_CONFIG`.

```python
{
   "BatchStrategy": "SingleRecord",
   "Environment": { 
      "INFERENCE_CONFIG" : "{ \"num_samples\": 200, \"output_types\": [\"mean\"] }",
      ...
   },
   "TransformInput": {
      "SplitType": "Line",
      ...
   },
   "TransformOutput": { 
      "AssembleWith": "Line",
      ...
   },
   ...
}
```
