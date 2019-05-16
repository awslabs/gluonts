# GluonTS - Time Series Forecasting in Python

## Development setup

Upon checking out this package, please run the following script.

```bash
./dev_setup.sh
```

## Build instructions

To build the project from source:

```bash
python setup.py build
```

To run the project tests:

```bash
python setup.py tests
```

To build the project documentation:

```bash
python setup.py docs
```

To view the project documentation after building it:

```bash
open docs/_build/html/index.html
```
 

## Development Container

To build the development container, go to the `container` folder and run the following commands:

```bash
image  # creates a local image called `gluonts/dev`
tag    # tags `gluonts/dev:latest` as `810547253217.dkr.ecr.us-west-2.amazonaws.com:gluonts/dev:latest`
push   # pushes `810547253217.dkr.ecr.us-west-2.amazonaws.com:gluonts/dev:latest`
```

You can use the pushed ECR image as a custom `image_name` for your SageMaker `train` jobs. 
To select the GluonTS `Estimator` estimator class to be used during training, add the fully-qualified `Estimator` class name to `estimator_class` to your training job hyperparameters.

## Models

Model in GluonTS are intended to run out of the box. For this, they should follow the convention to provide 
default hyper-parameters when possible. For every model, the only required parameters should be:

* `freq`: str
* `prediction_length`: int

Many parameters are used for different models such as `batch_size` or `num_layers`. The naming of parameters is unified 
across models based on the following conventions (names are selected to be as close as possible to SageMaker):

* `context_length`: Optional[PositiveInt] 
* `trainer`: Trainer
* `num_batches_per_epoch`: PositiveInt
* `num_layers`: PositiveInt
* `num_cells`: PositiveInt
* `batch_size`: PositiveInt
* `num_eval_samples`: PositiveInt
* `dropout_rate`: PositiveFloat
* `distribution`: DistributionFactory
* `hybridize`: bool

## Examples

The following modules are good entry-points to understand how to use GluonTS:

* `gluonts.example.run_simple_feedforward`: how to train and evaluate a model.
* `gluonts.example.benchmark`: how to evaluate and compare several model.
* `gluonts.model.seasonal_naive`: how to implement simple models using just NumPy and Pandas.
* `gluonts.model.simple_feedforward.estimator`: how to define a Gluon model.
