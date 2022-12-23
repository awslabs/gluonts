# TSBench

This repository provides the code for TSBench which, as far as we know, is the most comprehensive evaluation of forecasting
methods to this date. Please reference the following paper if you use the benchmark in your research:

**Multi-Objective Model Selection for Time Series Forecasting**. *Oliver Borchert, David Salinas, Valentin Flunkert, Tim Januschowski, Stephan Günnemann*. https://arxiv.org/abs/2202.08485

## Features

The code that is found in this repository provides the following features:

- It provides well-structured code to benchmark forecasting methods on AWS Sagemaker. The
  repository readily allows the evaluation of 13 time series forecasting methods included in
  GluonTS across 44 publicly available datasets. Evaluations are performed with respect to multiple
  performance measures and all forecasts generated for the test set are stored.
- The evaluations of these 13 forecasting methods on all 44 datasets are publicly available on the
  [Registry of Open Data on AWS](https://registry.opendata.aws/tsbench/).
- It allows to evaluate ensembles of models quickly by using the stored forecasts on the test sets.
- It enables to train surrogate models that learn from the performance of forecasting methods
  across datasets. The surrogate models then allow to select models and hyperparameters for unseen
  datasets while considering multiple objectives (e.g. accuracy and latency).

While we provide out-of-the-box scripts to schedule all datasets/methods combinations in parallel on SageMaker, the code should be easy to adapt on other environments given that we run in docker containers.
  
## Installation

Prior to installation, you may want to install all dependencies (Python, CUDA, Poetry). If you are
running on an AWS EC2 instance with Ubuntu 20.04, you can use the provided bash script:

```bash
bash bin/setup-ec2.sh
```

In order to use the code in this repository, you should first clone the GluonTS repository and then
go into the directory of this project:

```bash
git clone git@github.com:awslabs/gluonts.git
cd gluonts/src/gluonts/nursery/tsbench
```

Then, in the root of the repository, you can install all dependencies via
[Poetry](https://python-poetry.org):

```bash
poetry install
```

_Note: TSBench does not currently run on Apple Silicon devices as multiple dependencies are
unavailable._

## Command Line Interface

The main way of interaction with the code in this repository should be the `tsbench` CLI. After
running `poetry install` and `poetry shell` in the root of the TSBench repository, you can get an
overview by running the help command:

```bash
tsbench --help
```

In the following in-depth examples, we will also heavily use the CLI.

## Using Publicly Available Evaluations

As noted above, the evaluation of 13 forecasting methods (along with various hyperparameter
settings) across all 44 datasets included in this repository are publicly available on the
[Registry of Open Data on AWS](https://registry.opendata.aws/tsbench/). You can easily download
these evaluations using the CLI:

```bash
tsbench evaluations download
```

This command only downloads the performance metrics (i.e. accuracy, latency of forecasts, etc.) and
does not download the generated forecasts. This allows you to download only ~20 MiB of data. If you
actually want to access the forecasts, you can pass an additional flag which will download roughly
600 GiB of data:

```bash
tsbench evaluations download --include_forecasts
```

All data that is downloaded will be available at `~/evaluations`. You can customize this path by
setting `--evaluations_path`. This will require you to manually set this path when executing plenty
of other commands though.

The notebooks in the [examples](./examples) folder will guide you through the usage of locally
available evaluations. See below for more context on the content of these example notebooks.

## Running Evaluations

One of the main purposes of this repository is to easily benchmark forecasting methods on various
datasets. In the following, we want to guide you through the entire process.

### Set Up an EC2 Instance

First, you should set up an EC2 instance (since you will be dealing with plenty of network
traffic). In order to use all functionalities, make sure to attach the following permission
policies to its IAM account:

- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

On this EC2 instance, you should now clone the repository and install its dependencies.

### Prepare the Data

Before evaluating forecasting methods, you need to prepare the benchmark datasets. To download all
datasets that are bundled in TSBench, you need to have a Kaggle account and you need to add your
API token at `~/.kaggle/kaggle.json`. Prior to using the `tsbench` CLI, you have to download some
datasets via the Kaggle API. For this, run the following script (which potentially requires you to
go into your browser to accept terms of use):

```bash
bash bin/download-kaggle.sh
```

Afterwards, you can run the following commands (assuming that you have executed `poetry shell`):

```bash
# Download and preprocess all datasets
tsbench datasets download

# Upload locally available datasets to your S3 bucket
tsbench datasets upload --bucket <your_bucket_name>
```

Remember the name of the bucket that you used here. You will need it later!

### Prepare AWS Sagemaker

As evaluations are scheduled on AWS Sagemaker, you will need to ensure that the IAM account that
runs the evaluations can access the data in the bucket that you just created. For this, create an
IAM role which has (at least) the following policies attached:

- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

### Prepare your Docker Container

As training jobs on AWS Sagemaker run in Docker containers, you will need to build your own and
upload it to the ECR registry. For this, you must first create an ECR repository named `tsbench`.
Then, you can build and upload it by using the following utility script:

```bash
bash bin/build-container.sh
```

### Schedule Evaluations

At this point, you can finally schedule your evaluations. In the
[configs/benchmark](./configs/benchmark) directory, you will find the full grid search over methods
and datasets that are included in the publicly available evaluations. If you want to re-run them,
you can simply execute the following:

```bash
tsbench evaluations schedule \
    --config_path configs/benchmark \
    --sagemaker_role <arn_of_your_sagemaker_role> \
    --experiment <your_experiment_name> \
    --data_bucket <your_bucket_name> \
    --output_bucket <your_bucket_name>
```

This will schedule all evaluations on your AWS Sagemaker account and groups them by the experiment
name. After they have completed successfully, you can download them into your local evaluation
directory:

```bash
tsbench evaluations download --experiment <your_experiment_name>
```

Again, this only downloads the performance metrics. If you actually want to use the forecasts, add
the `--include_forecasts` flag.

_Note: As this command may run for a long time (depending on the number of evaluations you run and
your AWS Sagemaker quotas), it is a good idea to run this command in a `tmux` session._

### Extending the Benchmark

#### New Datasets

If you want to run evaluations for your own dataset, you can easily add it to the registry by
editing the [dataset definitions file](./tsbench/config/dataset/datasets.py). Consult the
implementation of the included datasets to get an idea of how to add your own.

By passing a unique key for your dataset to the `register` class decorator, your dataset is readily
usable across all CLI commands and scripts.

#### New Models

In order to use time series forecasting methods that are not included in this repository, you can
edit the [model definitions file](./tsbench/config/model/models.py). In case you attempt to use
newly published estimators from GluonTS, edit the dependencies in
[pyproject.toml](./pyproject.toml) to use a different GluonTS version.

If your new model defines non-standard hyperparameters (i.e. hyperparameters other than the
training time or the learning rate), you should add them as options to the
[evaluation script](./tsbench/evaluate.py). The option should take the form
`--<model_key>_<hyperparameter_name>`.

## Additional Examples

The [examples](./examples) directory provides additional usage examples of the code in this
repository:

- [`browse-offline-evaluations.ipynb`](./examples/browse-offline-evaluations.ipynb) explains how to
  access evaluations that have been downloaded
- [`train-a-recommender.ipynb`](./examples/train-a-recommender.ipynb`) discusses how you can train
  a recommender which is able to provide multi-objective recommendations of forecasting models and
  hyperparameters for unseen datasets
- [`evaluate-ensemble-performance.ipynb`](./examples/evaluate-ensemble-performance.ipynb) goes
  through the process of simulating the performance of an ensemble using locally available
  forecasts
- [`analyze-surrogate-performance.ipynb`](./examples/analyze-surrogate-performance.ipynb) shows how
  you can assess the performance of surrogate models which learn from the offline evaluations

## Citation

<tbd>
