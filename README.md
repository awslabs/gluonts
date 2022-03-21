# GluonTS - Probabilistic Time Series Modeling in Python

[stable docs url]: https://ts.gluon.ai/
[development docs url]: https://ts.gluon.ai/master/index.html

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

GluonTS requires Python 3.7, and the easiest way to install it is via `pip`:

```bash
pip install --upgrade gluonts mxnet~=1.8   # to be able to use MXNet-based models
pip install --upgrade gluonts torch~=1.10  # to be able to use PyTorch-based models
```

## Documentation

* [Documentation (stable version)][stable docs url]
* [Documentation (development version)][development docs url]
* [JMLR MLOSS Paper](http://www.jmlr.org/papers/v21/19-820.html)
* [ArXiv Paper](https://arxiv.org/abs/1906.05264)

## Available models

Name                             | Local/global | Data layout              | Architecture/method | Implementation | References
---------------------------------|--------------|--------------------------|---------------------|----------------|-----------
DeepAR                           | Global       | Univariate               | RNN | [MXNet](src/gluonts/model/deepar/_estimator.py), [PyTorch](src/gluonts/torch/model/deepar/estimator.py) | [paper](https://doi.org/10.1016/j.ijforecast.2019.07.001)
DeepState                        | Global       | Univariate               | RNN, state-space model | [MXNet](src/gluonts/model/deepstate/_estimator.py) | [paper](https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html)
DeepFactor                       | Global       | Univariate               | RNN, state-space model, Gaussian process | [MXNet](src/gluonts/model/deep_factor/_estimator.py) | [paper](https://proceedings.mlr.press/v97/wang19k.html)
Deep Renewal Processes           | Global       | Univariate               | RNN | [MXNet](src/gluonts/model/renewal/_estimator.py) | [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764)
GPForecaster                     | Global       | Univariate               | MLP, Gaussian process | [MXNet](src/gluonts/model/gp_forecaster/_estimator.py) | -
MQ-CNN                           | Global       | Univariate               | CNN encoder, MLP decoder | [MXNet](src/gluonts/model/seq2seq/_mq_dnn_estimator.py) | [paper](https://arxiv.org/abs/1711.11053)
MQ-RNN                           | Global       | Univariate               | RNN encoder, MLP encoder | [MXNet](src/gluonts/model/seq2seq/_mq_dnn_estimator.py) | [paper](https://arxiv.org/abs/1711.11053)
N-BEATS                          | Global       | Univariate               | MLP, residual links | [MXNet](https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/model/n_beats/_estimator.py) | [paper](https://openreview.net/forum?id=r1ecqn4YwB)
Rotbaum                          | Global       | Univariate               | XGBoost, Quantile Regression Forests, LightGBM, Level Set Forecaster | [Numpy](src/gluonts/model/rotbaum/_estimator.py) | [paper](https://openreview.net/forum?id=VD3TMzyxKK)
Causal Convolutional Transformer | Global       | Univariate               | Causal convolution, self attention | [MXNet](src/gluonts/model/san/_estimator.py) | [paper](https://papers.nips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html)
Temporal Fusion Transformer      | Global       | Univariate               | LSTM, self attention | [MXNet](src/gluonts/model/tft/_estimator.py) | [paper](https://doi.org/10.1016/j.ijforecast.2021.03.012)
Transformer                      | Global       | Univariate               | MLP, multi-head attention | [MXNet](src/gluonts/model/transformer/_estimator.py) | [paper](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
WaveNet                          | Global       | Univariate               | Dilated convolution | [MXNet](src/gluonts/model/wavenet/_estimator.py) | [paper](https://arxiv.org/abs/1609.03499)
SimpleFeedForward                | Global       | Univariate               | MLP | [MXNet](src/gluonts/model/simple_feedforward/_estimator.py), [PyTorch](src/gluonts/torch/model/simple_feedforward/estimator.py) | -
DeepVAR                          | Global       | Multivariate             | RNN | [MXNet](src/gluonts/model/deepvar/_estimator.py) | [paper](https://proceedings.neurips.cc/paper/2019/hash/0b105cf1504c4e241fcc6d519ea962fb-Abstract.html)
GPVAR                            | Global       | Multivariate             | RNN, Gaussian process | [MXNet](src/gluonts/model/gpvar/_estimator.py) | [paper](https://proceedings.neurips.cc/paper/2019/hash/0b105cf1504c4e241fcc6d519ea962fb-Abstract.html)
LSTNet                           | Global       | Multivariate             | LSTM | [MXNet](src/gluonts/model/lstnet/_estimator.py) | [paper](https://doi.org/10.1145/3209978.3210006)
DeepTPP                          | Global       | Multivariate events      | RNN, temporal point process | [MXNet](src/gluonts/model/tpp/deeptpp/_estimator.py) | [paper](https://arxiv.org/pdf/1909.12127)
RForecast                        | Local        | Univariate               | ARIMA, ETS, Croston, TBATS | [Wrapped R package](src/gluonts/model/r_forecast/_predictor.py) | [paper](https://www.jstatsoft.org/article/view/v027i03)
Prophet                          | Local        | Univariate               | - | [Wrapped Python package](src/gluonts/model/prophet/_predictor.py) | [paper](https://doi.org/10.1080/00031305.2017.1380080)
NaiveSeasonal                    | Local        | Univariate               | - | [Numpy](src/gluonts/model/seasonal_naive/_predictor.py) | [book section](https://otexts.com/fpp2/simple-methods.html#seasonal-na%C3%AFve-method)
Naive2                           | Local        | Univariate               | - | [Numpy](src/gluonts/model/naive_2/_predictor.py) | [book section](https://www.wiley.com/en-ie/Forecasting:+Methods+and+Applications,+3rd+Edition-p-9780471532330)
NPTS                             | Local        | Univariate               | - | [Numpy](src/gluonts/model/npts/_predictor.py) | -

## Running on Amazon SageMaker

Training and deploying GluonTS models on [Amazon SageMaker](https://aws.amazon.com/de/sagemaker/) is easily done by using the `gluonts.shell` package, see [its README](https://github.com/awslabs/gluon-ts/tree/master/src/gluonts/shell) for more information.
Dockerfiles compatible with Amazon SageMaker can be found in the [examples/dockerfiles](https://github.com/awslabs/gluon-ts/tree/master/examples/dockerfiles) folder.

## Quick example

This simple example illustrates how to train a model from GluonTS on some data, and then use it to make predictions.
For more extensive example, please refer to the [tutorial section of the documentation](https://ts.gluon.ai/tutorials/index.html)

As a first step, we need to collect some data: in this example we will use the volume of tweets mentioning the
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

## Contributing

If you wish to contribute to the project, please refer to our
[contribution guidelines](https://github.com/awslabs/gluon-ts/tree/master/CONTRIBUTING.md).

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

* [Collected Papers from the group behind GluonTS](https://github.com/awslabs/gluon-ts/tree/master/REFERENCES.md): a bibliography.
* [Tutorial at IJCAI 2021 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-IJCAI-2021/) with [YouTube link](https://youtu.be/AB3I9pdT46c). 
* [Tutorial at WWW 2020 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-WWW-2020/)
* [Tutorial at SIGMOD 2019](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
* [Tutorial at KDD 2019](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
* [Tutorial at VLDB 2018](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)
* [Neural Time Series with GluonTS](https://youtu.be/beEJMIt9xJ8)
* [International Symposium of Forecasting: Deep Learning for Forecasting workshop](https://lostella.github.io/ISF-2020-Deep-Learning-Workshop/)