# GluonTS - Probabilistic Time Series Modeling in Python

[stable docs url]: https://ts.gluon.ai/
[development docs url]: https://ts.gluon.ai/dev/index.html

[![PyPI](https://img.shields.io/pypi/v/gluonts.svg?style=flat-square)](https://pypi.org/project/gluonts/)
[![GitHub](https://img.shields.io/github/license/awslabs/gluon-ts.svg?style=flat-square)](./LICENSE)
[![Static](https://img.shields.io/static/v1?label=docs&message=stable&color=blue&style=flat-square)][stable docs url]
[![Static](https://img.shields.io/static/v1?label=docs&message=dev&color=blue&style=flat-square)][development docs url]
[![PyPI Downloads](https://pepy.tech/badge/gluonts)](https://pypi.org/project/gluonts/)

GluonTS is a Python package for probabilistic time series modeling, focusing on deep learning based models.

## Features

* State-of-the-art models implemented with [MXNet](https://mxnet.incubator.apache.org/) and [PyTorch](https://pytorch.org/) (see [list](#available-models))
* Easy AWS integration via [Amazon SageMaker](https://aws.amazon.com/de/sagemaker/) (see [here](#running-on-amazon-sagemaker))
* Utilities for loading and iterating over time series datasets
* Utilities to evaluate models performance and compare their accuracy
* Building blocks to define custom models and quickly experiment

## Installation

GluonTS requires Python 3.6 or newer, and the easiest way to install it is via `pip`:

```bash
pip install gluonts[mxnet,pro]  # support for mxnet models, faster datasets
pip install gluonts[torch,pro]  # support for torch models, faster datasets
```

You can enable or disable extra dependencies as you prefer, depending on what GluonTS features you are interested in enabling.

**Models:**

* `mxnet` - MXNet-based models
* `torch` - PyTorch-based models
* `R` - R-based models
* `Prophet` - Prophet-based models

**Datasets:**

* `arrow` - Arrow and Parquet dataset support
* `pro` - bundles `arrow` plus `orjson` for faster datasets

**Misc:**

* `shell` for integration with SageMaker


## Documentation

* [Documentation (stable version)][stable docs url]
* [Documentation (development version)][development docs url]
* [JMLR MLOSS Paper](http://www.jmlr.org/papers/v21/19-820.html)
* [ArXiv Paper](https://arxiv.org/abs/1906.05264)



## Running on Amazon SageMaker

Training and deploying GluonTS models on [Amazon SageMaker](https://aws.amazon.com/de/sagemaker/) is easily done by using the `gluonts.shell` package, see [its README](https://github.com/awslabs/gluon-ts/tree/dev/src/gluonts/shell) for more information.
Dockerfiles compatible with Amazon SageMaker can be found in the [examples/dockerfiles](https://github.com/awslabs/gluon-ts/tree/dev/examples/dockerfiles) folder.

## Quick example

This simple example illustrates how to train a model from GluonTS on some data, and then use it to make predictions.
For more extensive example, please refer to the [tutorial section of the documentation](https://ts.gluon.ai/tutorials/index.html)

As a first step, we need to collect some data: in this example we will use the volume of tweets mentioning the
AMZN ticker symbol.

```python
import pandas as pd
url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df = pd.read_csv(
    url,
    index_col=0,
    parse_dates=True,
    header=0,
    names=["target"],
)
```

The first 100 data points look like follows:

```python
import matplotlib.pyplot as plt
df[:100].plot(linewidth=2)
plt.grid(which='both')
plt.show()
```

![Data](https://github.com/awslabs/gluon-ts/raw/dev/docs/figures/Tweets_AMZN_data.png)

We can now prepare a training dataset for our model to train on.
Datasets in GluonTS are essentially iterable collections of
dictionaries: each dictionary represents a time series
with possibly associated features. For this example, we only have one
entry, specified by the `"start"` field which is the timestamp of the
first datapoint, and the `"target"` field containing time series data.
For training, we will use data up to midnight on April 5th, 2015.

```python
from gluonts.dataset.pandas import PandasDataset

training_data = PandasDataset(df[:"2015-04-05 00:00:00"])
```

A forecasting model in GluonTS is a *predictor* object. One way of obtaining
predictors is by training a correspondent *estimator*. Instantiating an
estimator requires specifying the frequency of the time series that it will
handle, as well as the number of time steps to predict. In our example
we're using 5 minutes data, so `freq="5min"`,
and we will train a model to predict the next hour, so `prediction_length=12`.
We also specify some minimal training options.

```python
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

estimator = DeepAREstimator(freq="5min", prediction_length=12, trainer=Trainer(epochs=10))
predictor = estimator.train(training_data=training_data)
```

During training, useful information about the progress will be displayed.
To get a full overview of the available options, please refer to the
documentation of `DeepAREstimator` (or other estimators) and `Trainer`.

We're now ready to make predictions: we will forecast the hour following
the midnight on April 15th, 2015.

```python
test_data = PandasDataset(df[:"2015-04-15 00:00:00"])

from gluonts.dataset.util import to_pandas

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
```

![Forecast](https://github.com/awslabs/gluon-ts/raw/dev/docs/figures/Tweets_AMZN_forecast.png)

Note that the forecast is displayed in terms of a probability distribution:
the shaded areas represent the 50% and 90% prediction intervals, respectively,
centered around the median (dark green line).

## Contributing

If you wish to contribute to the project, please refer to our
[contribution guidelines](https://github.com/awslabs/gluon-ts/tree/dev/CONTRIBUTING.md).

## Citing

If you use GluonTS in a scientific publication, we encourage you to add the following references to the related papers,
in addition to any model-specific references that are relevant for your work:

```bibtex
@article{gluonts_jmlr,
  author  = {Alexander Alexandrov and Konstantinos Benidis and Michael Bohlke-Schneider
    and Valentin Flunkert and Jan Gasthaus and Tim Januschowski and Danielle C. Maddix
    and Syama Rangapuram and David Salinas and Jasper Schulz and Lorenzo Stella and
    Ali Caner Türkmen and Yuyang Wang},
  title   = {{GluonTS: Probabilistic and Neural Time Series Modeling in Python}},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {116},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v21/19-820.html}
}
```

```bibtex
@article{gluonts_arxiv,
  author  = {Alexandrov, A. and Benidis, K. and Bohlke-Schneider, M. and
    Flunkert, V. and Gasthaus, J. and Januschowski, T. and Maddix, D. C.
    and Rangapuram, S. and Salinas, D. and Schulz, J. and Stella, L. and
    Türkmen, A. C. and Wang, Y.},
  title   = {{GluonTS: Probabilistic Time Series Modeling in Python}},
  journal = {arXiv preprint arXiv:1906.05264},
  year    = {2019}
}
```

## Other resources

* [Collected Papers from the group behind GluonTS](https://github.com/awslabs/gluon-ts/tree/dev/REFERENCES.md): a bibliography.
* [Tutorial at IJCAI 2021 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-IJCAI-2021/) with [YouTube link](https://youtu.be/AB3I9pdT46c). 
* [Tutorial at WWW 2020 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-WWW-2020/)
* [Tutorial at SIGMOD 2019](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
* [Tutorial at KDD 2019](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
* [Tutorial at VLDB 2018](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)
* [Neural Time Series with GluonTS](https://youtu.be/beEJMIt9xJ8)
* [International Symposium of Forecasting: Deep Learning for Forecasting workshop](https://lostella.github.io/ISF-2020-Deep-Learning-Workshop/)
