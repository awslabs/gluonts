# GluonTS - Probabilistic Time Series Modeling in Python

[![PyPI](https://img.shields.io/pypi/v/gluonts.svg?style=flat-square)](https://pypi.org/project/gluonts/)
[![GitHub](https://img.shields.io/github/license/awslabs/gluon-ts.svg?style=flat-square)](./LICENSE)
[![Static](https://img.shields.io/static/v1?label=docs&message=stable&color=blue&style=flat-square)][stable docs url]
[![Static](https://img.shields.io/static/v1?label=docs&message=latest&color=blue&style=flat-square)][latest docs url]

GluonTS is a Python toolkit for probabilistic time series modeling,
built around [Apache MXNet (incubating)](https://mxnet.incubator.apache.org/).

GluonTS provides utilities for loading and iterating over time series datasets,
state of the art models ready to be trained, and building blocks to define
your own models and quickly experiment with different solutions.

* [Documentation (stable version)][stable docs url]
* [Documentation (latest)][latest docs url]
* [JMLR MLOSS Paper](http://www.jmlr.org/papers/v21/19-820.html)
* [ArXiv Paper](https://arxiv.org/abs/1906.05264)

[stable docs url]: https://gluon-ts.mxnet.io/
[latest docs url]: https://gluon-ts.s3-accelerate.dualstack.amazonaws.com/master/index.html

## Installation

GluonTS requires Python 3.6, and the easiest
way to install it is via `pip`:

```bash
pip install --upgrade mxnet~=1.7 gluonts
```

## Dockerfiles

Dockerfiles compatible with Amazon Sagemaker can be found in the [examples/dockerfiles](https://github.com/awslabs/gluon-ts/tree/master/examples/dockerfiles) folder.

## Quick start guide

This simple example illustrates how to train a model from GluonTS on some data,
and then use it to make predictions. As a first step, we need to collect
some data: in this example we will use the volume of tweets mentioning the
AMZN ticker symbol.

```python
import pandas as pd
url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df = pd.read_csv(url, header=0, index_col=0)
```

The first 100 data points look like follows:

```python
import matplotlib.pyplot as plt
df[:100].plot(linewidth=2)
plt.grid(which='both')
plt.show()
```

![Data](https://github.com/awslabs/gluon-ts/raw/master/docs/figures/Tweets_AMZN_data.png)

We can now prepare a training dataset for our model to train on.
Datasets in GluonTS are essentially iterable collections of
dictionaries: each dictionary represents a time series
with possibly associated features. For this example, we only have one
entry, specified by the `"start"` field which is the timestamp of the
first datapoint, and the `"target"` field containing time series data.
For training, we will use data up to midnight on April 5th, 2015.

```python
from gluonts.dataset.common import ListDataset
training_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
    freq = "5min"
)
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
test_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-15 00:00:00"]}],
    freq = "5min"
)

from gluonts.dataset.util import to_pandas

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
```

![Forecast](https://github.com/awslabs/gluon-ts/raw/master/docs/figures/Tweets_AMZN_forecast.png)

Note that the forecast is displayed in terms of a probability distribution:
the shaded areas represent the 50% and 90% prediction intervals, respectively,
centered around the median (dark green line).

## Further examples

The following are good entry-points to understand how to use
many features of GluonTS:

* [Quick Start Tutorial](https://gluon-ts.mxnet.io/examples/basic_forecasting_tutorial/tutorial.html#Quick-Start-Tutorial): a quick start guide.
* [Extended Forecasting Tutorial](https://gluon-ts.mxnet.io/examples/extended_forecasting_tutorial/extended_tutorial.html): a detailed tutorial on forecasting using GluonTS.
* [evaluate_model.py](https://github.com/awslabs/gluon-ts/tree/master/examples/evaluate_model.py): how to train a model and compute evaluation metrics.
* [benchmark_m4.py](https://github.com/awslabs/gluon-ts/tree/master/examples/benchmark_m4.py): how to evaluate and compare multiple models on multiple datasets.

The following modules illustrate how custom models can be implemented:

* [`gluonts.model.seasonal_naive`](https://github.com/awslabs/gluon-ts/tree/master/src/gluonts/model/seasonal_naive): how to implement simple models using just NumPy and Pandas.
* [`gluonts.model.simple_feedforward`](https://github.com/awslabs/gluon-ts/tree/master/src/gluonts/model/simple_feedforward): how to define a trainable, Gluon-based model.

## Contributing

If you wish to contribute to the project, please refer to our
[contribution guidelines](https://github.com/awslabs/gluon-ts/tree/master/CONTRIBUTING.md).

## Citing

If you use GluonTS in a scientific publication, we encourage you to add
the following references to the related papers:

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

## Video
* [Neural Time Series with GluonTS](https://youtu.be/beEJMIt9xJ8)

## Further Reading 

* [Collected Papers from the group behind GluonTS](https://github.com/awslabs/gluon-ts/tree/master/REFERENCES.md): a bibliography.

### Overview tutorials
* [Tutorial at WWW 20202 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-WWW-2020/)
* [Tutorial at SIGMOD 2019](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
* [Tutorial at KDD 2019](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
* [Tutorial at VLDB 2018](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)

### Introductory material
* [International Symposium of Forecasting: Deep Learning for Forecasting workshop](https://lostella.github.io/ISF-2020-Deep-Learning-Workshop/)
