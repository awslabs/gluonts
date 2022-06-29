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
N-BEATS                          | Global       | Univariate               | MLP, residual links | [MXNet](src/gluonts/model/n_beats/_estimator.py) | [paper](https://openreview.net/forum?id=r1ecqn4YwB)
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
