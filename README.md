<img class="hide-on-website" height="100px" src="https://ts.gluon.ai/dev/_static/gluonts.svg">

# GluonTS - Probabilistic Time Series Modeling in Python

[![PyPI](https://img.shields.io/pypi/v/gluonts.svg?style=flat-square&color=b75347)](https://pypi.org/project/gluonts/)
[![GitHub](https://img.shields.io/github/license/awslabs/gluonts.svg?style=flat-square&color=df7e66)](./LICENSE)
[![Static](https://img.shields.io/static/v1?label=docs&message=stable&color=edc775&style=flat-square)](https://ts.gluon.ai/)
[![Static](https://img.shields.io/static/v1?label=docs&message=dev&color=edc775&style=flat-square)](https://ts.gluon.ai/dev/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/gluonts?style=flat-square&color=94b594)](https://pepy.tech/project/gluonts)

GluonTS is a Python package for probabilistic time series modeling, focusing on deep learning based models,
based on [PyTorch](https://pytorch.org) and [MXNet](https://mxnet.apache.org).


## Installation

GluonTS requires Python 3.6 or newer, and the easiest way to install it is via `pip`:

```bash
# support for mxnet models, faster datasets
pip install "gluonts[mxnet,pro]"

# support for torch models, faster datasets
pip install "gluonts[torch,pro]"
```

## Simple Example

To illustrate how to use GluonTS, we train a DeepAR-model and make predictions
using the simple "airpassengers" dataset. The dataset consists of a single
time series, containing monthly international passengers between the years
1949 and 1960, a total of 144 values (12 years * 12 months). We split the
dataset into train and test parts, by removing the last three years (36 month)
from the train data. Thus, we will train a model on just the first nine years
of data.


```py
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.mx import DeepAREstimator, Trainer

# Load data from a CSV file into a PandasDataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/"
    "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
    index_col=0,
    parse_dates=True,
)
dataset = PandasDataset(df, target="#Passengers")

# Train a DeepAR model on all data but the last 36 months
training_data, test_gen = split(dataset, offset=-36)
model = DeepAREstimator(
    prediction_length=12, freq="M", trainer=Trainer(epochs=5)
).train(training_data)

# Generate test instances and predictions for them
test_data = test_gen.generate_instances(prediction_length=12, windows=3)
forecasts = list(model.predict(test_data.input))

# Plot predictions
df["#Passengers"].plot(color="black")
for forecast, color in zip(forecasts, ["green", "blue", "purple"]):
    forecast.plot(color=f"tab:{color}")
plt.legend(["True values"], loc="upper left", fontsize="xx-large")
```

![[train-test]](https://d2kv9n23y3w0pn.cloudfront.net/static/README/forecasts.png)


Note that the forecasts are displayed in terms of a probability distribution:
The shaded areas represent the 50% and 90% prediction intervals, respectively,
centered around the median.

## Contributing

If you wish to contribute to the project, please refer to our
[contribution guidelines](https://github.com/awslabs/gluonts/tree/dev/CONTRIBUTING.md).

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

## Links

### Documentation

* [Documentation (stable)](https://ts.gluon.ai/stable/)
* [Documentation (development)](https://ts.gluon.ai/dev/)

### References

* [JMLR MLOSS Paper](http://www.jmlr.org/papers/v21/19-820.html)
* [ArXiv Paper](https://arxiv.org/abs/1906.05264)
* [Collected Papers from the group behind GluonTS](https://github.com/awslabs/gluonts/tree/dev/REFERENCES.md): a bibliography.

### Tutorials and Workshops

* [Tutorial at IJCAI 2021 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-IJCAI-2021/) with [YouTube link](https://youtu.be/AB3I9pdT46c). 
* [Tutorial at WWW 2020 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-WWW-2020/)
* [Tutorial at SIGMOD 2019](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
* [Tutorial at KDD 2019](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
* [Tutorial at VLDB 2018](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)
* [Neural Time Series with GluonTS](https://youtu.be/beEJMIt9xJ8)
* [International Symposium of Forecasting: Deep Learning for Forecasting workshop](https://lostella.github.io/ISF-2020-Deep-Learning-Workshop/)
