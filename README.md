# GluonTS - Probabilistic Time Series Modeling in Python

[stable docs url]: https://ts.gluon.ai/
[development docs url]: https://ts.gluon.ai/dev/index.html

[![PyPI](https://img.shields.io/pypi/v/gluonts.svg?style=flat-square)](https://pypi.org/project/gluonts/)
[![GitHub](https://img.shields.io/github/license/awslabs/gluon-ts.svg?style=flat-square)](./LICENSE)
[![Static](https://img.shields.io/static/v1?label=docs&message=stable&color=blue&style=flat-square)][stable docs url]
[![Static](https://img.shields.io/static/v1?label=docs&message=dev&color=blue&style=flat-square)][development docs url]
[![PyPI Downloads](https://pepy.tech/badge/gluonts)](https://pypi.org/project/gluonts/)

GluonTS is a Python package for probabilistic time series modeling, focusing on deep learning based models.


## Installation

GluonTS requires Python 3.6 or newer, and the easiest way to install it is via `pip`:

```bash
# support for mxnet models, faster datasets
pip install gluonts[mxnet,pro]  

# support for torch models, faster datasets
pip install gluonts[torch,pro]
```


## Documentation

* [Documentation (stable version)][stable docs url]
* [Documentation (development version)][development docs url]
* [JMLR MLOSS Paper](http://www.jmlr.org/papers/v21/19-820.html)
* [ArXiv Paper](https://arxiv.org/abs/1906.05264)

## Features

* State-of-the-art models implemented with [MXNet](https://mxnet.incubator.apache.org/) and [PyTorch](https://pytorch.org/) (see [list](/blob/dev/docs/docs/getting_started/models.md))
* Easy AWS integration via [Amazon SageMaker](https://aws.amazon.com/de/sagemaker/) (see [here](#running-on-amazon-sagemaker))
* Utilities for loading and iterating over time series datasets
* Utilities to evaluate models performance and compare their accuracy
* Building blocks to define custom models and quickly experiment



## Quick example

To illustrate how to use GluonTS, we train a DeepAR-model and make predictions
using the simple "airpassengers" dataset. The dataset consists of a single
time-series, containing monthly international passengers between the years
1949 and 1960, a total of 144 values (12 years * 12 months). We split the
dataset into train and test parts, by removing the last three years (36 month)
from the train data. Thus, we will train a model on just the first nine years
of data

Peak into data:

```csv
# Month,#Passengers
# 1949-01,112
# 1949-02,118
# 1949-03,132
...
```

Loading, splitting and plotting of data:

```py
import pandas as pd

URL = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"

df = pd.read_csv(URL, index_col=0, parse_dates=True)

# Cut off three years for training
train = df[:-36]
# For testing, we take entire time-series
test = df

plt.plot(train.index, train.values, label="Train")
plt.plot(test.index, test.values, label="Test")
plt.legend(loc="upper left")
```

![[train-test]](https://ts.gluon.ai/static/README/train-test.png)

Train a model using `DeepAR`. We wrap the dataframes into instances of
`PandasDataset` and indicate that we want to use the `#Passengers` column as
the prediction target:

```py
from gluonts.dataset.pandas import PandasDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer

train_dataset = PandasDataset(train, target="#Passengers")

deepar = DeepAREstimator(
    # We want to predict one year at a time.
    prediction_length=12,
    # Use features for monthly-frequencies
    freq="M",
    # Train for just a few epochs to prevent overfitting.
    trainer=Trainer(epochs=5),
)
model = deepar.train(train_dataset)
```

Now we can use the model to make predictions by asking the model to forecast
each of the three years of our test dataset:

```py
prediction_input = PandasDataset(
    [
        test[:-36],
        test[:-24],
        test[:-12],
    ],
    target="#Passengers",
)

predictions = model.predict(prediction_input)


plt.plot(test.index, test.values, color="k")

for color, prediction in zip(["green", "blue", "purple"], predictions):
    prediction.plot(color=f"tab:{color}")


plt.legend(["True values"], loc="upper left", fontsize="xx-large")
```

![[train-test]](https://ts.gluon.ai/static/README/forecasts.png)


Note that the forecasts are displayed in terms of a probability distribution:
The shaded areas represent the 50% and 90% prediction intervals, respectively,
centered around the median.

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
