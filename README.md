# GluonTS - Probabilistic Time Series Modeling in Python

![PyPI](https://img.shields.io/pypi/v/gluonts.svg?style=flat-square) ![GitHub](https://img.shields.io/github/license/awslabs/gluon-ts.svg?style=flat-square)

GluonTS is a Python toolkit for probabilistic time series modeling,
built around [Apache MXNet (incubating)](https://mxnet.incubator.apache.org/).

GluonTS provides utilities for loading and iterating over time series datasets,
state of the art models ready to be trained, and building blocks to define
your own models and quickly experiment with different solutions.

* [Documentation](https://gluon-ts.mxnet.io/)
* [Paper](https://arxiv.org/abs/1906.05264)

## Installation

GluonTS requires Python 3.6, and the easiest
way to install it is via `pip`:

```bash
pip install gluonts
```

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
from gluonts.trainer import Trainer

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

* [Quick Start Tutorial](https://github.com/awslabs/gluon-ts/tree/master/docs/examples/basic_forecasting_tutorial/tutorial.md): a quick start guide.
* [Extended Forecasting Tutorial](https://github.com/awslabs/gluon-ts/tree/master/docs/examples/extended_forecasting_tutorial/extended_tutorial.md): a detailed tutorial on forecasting.
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
the following reference to the associated
[paper](https://arxiv.org/abs/1906.05264):

```
@article{gluonts,
  title={{GluonTS: Probabilistic Time Series Modeling in Python}},
  author={Alexandrov, A. and Benidis, K. and Bohlke-Schneider, M. and
          Flunkert, V. and Gasthaus, J. and Januschowski, T. and Maddix, D. C.
          and Rangapuram, S. and Salinas, D. and Schulz, J. and Stella, L. and
          Türkmen, A. C. and Wang, Y.},
  journal={arXiv preprint arXiv:1906.05264},
  year={2019}
}
```

## Further Papers

We encourage you to also check out work by the group behind 
GluonTS. They are grouped according to topic and ordered 
chronographically.

### Methods
A number of the below methods are available in GluonTS.

Deep Factor models, a global-local forecasting method.
```
@inproceedings{wang2019deepfactors,
	Author = {Wang, Yuyang and Smola, Alex and Maddix, Danielle and Gasthaus, Jan and Foster, Dean and Januschowski, Tim},
	Booktitle = {International Conference on Machine Learning},
	Pages = {6607--6617},
	Title = {Deep factors for forecasting},
	Year = {2019}
}
```
DeepAR, an RNN-based probabilistic forecasting model.
```
@article{flunkert2019deepar,
	Author = {Salinas, David and Flunkert, Valentin and Gasthaus, Jan and Tim Januschowski},
	Journal = {International Journal of Forecasting},
	Title = {DeepAR: Probabilistic forecasting with autoregressive recurrent networks},
	Year = {2019}
}
```
A flexible way to model probabilistic forecasts via spline quantile forecasts.
```
@inproceedings{gasthaus2019probabilistic,
	Author = {Gasthaus, Jan and Benidis, Konstantinos and Wang, Yuyang and Rangapuram, Syama Sundar and Salinas, David and Flunkert, Valentin and Januschowski, Tim},
	Booktitle = {The 22nd International Conference on Artificial Intelligence and Statistics},
	Date-Added = {2019-06-26 13:23:32 +0000},
	Date-Modified = {2019-06-26 13:24:07 +0000},
	Pages = {1901--1910},
	Title = {Probabilistic Forecasting with Spline Quantile Function RNNs},
	Year = {2019}
}
```
Using RNNs to parametrize State Space Models.
```
@inproceedings{rangapuram2018deep,
	Author = {Rangapuram, Syama Sundar and Seeger, Matthias W and Gasthaus, Jan and Stella, Lorenzo and Wang, Yuyang and Januschowski, Tim},
	Booktitle = {Advances in Neural Information Processing Systems},
	Date-Added = {2019-06-26 13:38:03 +0000},
	Date-Modified = {2019-06-26 13:38:43 +0000},
	Pages = {7785--7794},
	Title = {Deep state space models for time series forecasting},
	Year = {2018}
}
```
A scalable state space model.
```
@inproceedings{seeger2016bayesian,
	Author = {Seeger, Matthias W and Salinas, David and Flunkert, Valentin},
	Booktitle = {Advances in Neural Information Processing Systems},
	Date-Added = {2019-06-27 13:17:25 +0000},
	Date-Modified = {2019-06-27 13:18:01 +0000},
	Pages = {4646--4654},
	Title = {Bayesian intermittent demand forecasting for large inventories},
	Year = {2016}
}
```



### Tutorials
Tutorials are available in bibtex and with accompanying material,
 in particular slides, linked from below.
[Tutorial at KDD 2019](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
```
@inproceedings{faloutsos19forecasting2,
  author    = {Faloutsos, Christos and
               Flunkert, Valentin and
               Gasthaus, Jan and
               Januschowski, Tim and
               Wang, Yuyang},
  title     = {Forecasting Big Time Series: Theory and Practice},
  booktitle = {Proceedings of the 25th {ACM} {SIGKDD} International Conference on
               Knowledge Discovery {\&} Data Mining, {KDD} 2019, Anchorage, AK,
               USA, August 4-8, 2019.},
  year      = {2019}
  }
```
[Tutorial at SIGMOD 2019](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
```
@inproceedings{faloutsos2019forecasting,
 author = {Faloutsos, Christos and Gasthaus, Jan and Januschowski, Tim and Wang, Yuyang},
 title = {Classical and Contemporary Approaches to Big Time Series Forecasting},
 booktitle = {Proceedings of the 2019 International Conference on Management of Data},
 series = {SIGMOD '19},
 publisher = {ACM},
 address = {New York, NY, USA},
 year = {2019}
} 
```
[Tutorial at VLDB 2018](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)
```
@article{faloutsos2018forecasting,
	Author = {Faloutsos, Christos and Gasthaus, Jan and Januschowski, Tim and Wang, Yuyang},
	Date-Added = {2019-07-24 13:47:16 +0000},
	Date-Modified = {2019-07-24 13:48:00 +0000},
	Journal = {Proceedings of the VLDB Endowment},
	Number = {12},
	Pages = {2102--2105},
	Title = {Forecasting big time series: old and new},
	Volume = {11},
	Year = {2018}
}
```

### General audience
An overview of forecasting libraries in Python.
```
@article{januschowski19opensource,
  title={Open-Source Forecasting Tools in Python},
  author={Januschowski, Tim and Gasthaus, Jan and Wang, Yuyang},
  journal={Foresight: The International Journal of Applied Forecasting},
  year={2019}
}
```
A commentary on the M4 competition and its classification of the participating methods 
into 'statistical' and 'ML' methods. The article proposes alternative criteria.
```
@article{januschowski19criteria,
title = "Criteria for classifying forecasting methods",
author = "Januschowski, Tim and Gasthaus, Jan and  Wang, Yuyang and Salinas, David and Flunkert, Valentin and Bohlke-Schneider, Michael and Callot, Laurent"
journal = "International Journal of Forecasting",
year = "2019"
}
```
The business forecasting problem landscape can be divided into 
strategic, tactical and operational forecasting problems.
```
@article{januschowski18classification,
  title={A Classification of Business Forecasting Problems},
  author={Januschowski, Tim and Kolassa, Stephan},
  journal={Foresight: The International Journal of Applied Forecasting},
  year={2019},
  volume={52}, 
  pages={36-43}
}
```
A two-part article introducing deep learning for forecasting.
```
@article{januschowski18deeplearning2,
title = {Deep Learning for Forecasting: Current Trends and Challenges},
journal = {Foresight: The International Journal of Applied Forecasting},
year = "2018",
author = {Januschowski, Tim and Gasthaus, Jan and Wang, Yuyang and Rangapuram, Syama Sundar and Callot, Laurent},
volume = {51}, 
pages = {42-47}
}
```
```
@article{januschowski18deeplearning,
  title = {Deep Learning for Forecasting},
  author = {Januschowski, Tim and Gasthaus, Jan and Wang, Yuyang and Rangapuram, Syama and Callot, Laurent},
  journal = {Foresight},
  year = {2018}
}
```

### System Aspects
A large-scale retail forecasting system.
```
@article{bose2017probabilistic,
	Author = {B{\"o}se, Joos-Hendrik and Flunkert, Valentin and Gasthaus, Jan and Januschowski, Tim and Lange, Dustin and Salinas, David and Schelter, Sebastian and Seeger, Matthias and Wang, Yuyang},
	Date-Added = {2019-06-27 14:12:57 +0000},
	Date-Modified = {2019-06-27 14:13:35 +0000},
	Journal = {Proceedings of the VLDB Endowment},
	Number = {12},
	Pages = {1694--1705},
	Title = {Probabilistic demand forecasting at scale},
	Volume = {10},
	Year = {2017}
}
```