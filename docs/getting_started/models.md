## Available models

Model                                                       | Local/global | Data layout              | Architecture/method | Implementation
------------------------------------------------------------|--------------|--------------------------|---------------------|----------------
DeepAR [[Salinas et al. 2020][Salinas2020]]                 | Global       | Univariate               | RNN | [MXNet][DeepAR_mx], [PyTorch][DeepAR_torch]
DeepState [[Rangapuram et al. 2018][Rangapuram2018]]        | Global       | Univariate               | RNN, state-space model | [MXNet][DeepState]
DeepFactor [[Wang et al. 2019][Wang2019]]                   | Global       | Univariate               | RNN, state-space model, Gaussian process | [MXNet][DeepFactor]
Deep Renewal Processes [[Türkmen et al. 2021][Turkmen2021]] | Global       | Univariate               | RNN | [MXNet][DeepRenewal]
GPForecaster                                                | Global       | Univariate               | MLP, Gaussian process | [MXNet][GP]
MQ-CNN [[Wen et al. 2017][Wen2017]]                         | Global       | Univariate               | CNN encoder, MLP decoder | [MXNet][MQDNN]
MQ-RNN [[Wen et al. 2017][Wen2017]]                         | Global       | Univariate               | RNN encoder, MLP encoder | [MXNet][MQDNN]
N-BEATS [[Oreshkin et al. 2019][Oreshkin2019]]              | Global       | Univariate               | MLP, residual links | [MXNet][NBeats]
Rotbaum [[Hasson et al. 2021][Hasson2021]]                  | Global       | Univariate               | XGBoost, Quantile Regression Forests, LightGBM, Level Set Forecaster | [Numpy][Rotbaum]
Causal Convolutional Transformer [[Li et al. 2019][Li2019]] | Global       | Univariate               | Causal convolution, self attention | [MXNet][SAN]
Temporal Fusion Transformer [[Lim et al. 2021][Lim2021]]    | Global       | Univariate               | LSTM, self attention | [MXNet][TFT]
Transformer [[Vaswani et al. 2017][Vaswani2017]]            | Global       | Univariate               | MLP, multi-head attention | [MXNet][Transformer]
WaveNet [[van den Oord et al. 2016][vanDenOord2016]]        | Global       | Univariate               | Dilated convolution | [MXNet][WaveNet]
SimpleFeedForward                                           | Global       | Univariate               | MLP | [MXNet][SFF_mx], [PyTorch][SFF_torch]
DeepVAR [[Salinas et al. 2019][Salinas2019]]                | Global       | Multivariate             | RNN | [MXNet][DeepVAR]
GPVAR [[Salinas et al. 2019][Salinas2019]]                  | Global       | Multivariate             | RNN, Gaussian process | [MXNet][GPVAR]
LSTNet [[Lai et al. 2018][Lai2018]]                         | Global       | Multivariate             | LSTM | [MXNet][LSTNet]
DeepTPP [[Shchur et al. 2020][Shchur2020]]                  | Global       | Multivariate events      | RNN, temporal point process | [MXNet][DeepTPP]
RForecast [[Hyndman et al. 2008][Hyndman2008]]              | Local        | Univariate               | ARIMA, ETS, Croston, TBATS | [Wrapped R package][RForecast]
Prophet [[Taylor et al. 2017][Taylor2017]]                  | Local        | Univariate               | - | [Wrapped Python package][Prophet]
NaiveSeasonal [[Hyndman et al. 2018][Hyndman2018]]          | Local        | Univariate               | - | [Numpy][NaiveSeasonal]
Naive2 [[Makridakis et al. 1998][Makridakis1998]]           | Local        | Univariate               | - | [Numpy][Naive2]
NPTS                                                        | Local        | Univariate               | - | [Numpy][NPTS]

<!-- Links to bibliography -->

[Salinas2020]: https://doi.org/10.1016/j.ijforecast.2019.07.001
[Rangapuram2018]: https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
[Wang2019]: https://proceedings.mlr.press/v97/wang19k.html
[Turkmen2021]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764
[Wen2017]: https://arxiv.org/abs/1711.11053
[Oreshkin2019]: https://openreview.net/forum?id=r1ecqn4YwB
[Hasson2021]: https://openreview.net/forum?id=VD3TMzyxKK
[Li2019]: https://papers.nips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html
[Lim2021]: https://doi.org/10.1016/j.ijforecast.2021.03.012
[Vaswani2017]: https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
[vanDenOord2016]: https://arxiv.org/abs/1609.03499
[Salinas2019]: https://proceedings.neurips.cc/paper/2019/hash/0b105cf1504c4e241fcc6d519ea962fb-Abstract.html
[Lai2018]: https://doi.org/10.1145/3209978.3210006
[Shchur2020]: https://arxiv.org/pdf/1909.12127
[Hyndman2008]: https://www.jstatsoft.org/article/view/v027i03
[Taylor2017]: https://doi.org/10.1080/00031305.2017.1380080
[Hyndman2018]: https://otexts.com/fpp2/simple-methods.html#seasonal-na%C3%AFve-method
[Makridakis1998]: https://www.wiley.com/en-ie/Forecasting:+Methods+and+Applications,+3rd+Edition-p-9780471532330

<!-- Links to code -->

[DeepAR_mx]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/deepar/_estimator.py
[DeepAR_torch]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/torch/model/deepar/estimator.py
[DeepState]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/deepstate/_estimator.py
[DeepFactor]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/deep_factor/_estimator.py
[DeepRenewal]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/renewal/_estimator.py
[GP]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/gp_forecaster/_estimator.py
[MQDNN]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/seq2seq/_mq_dnn_estimator.py
[NBeats]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/n_beats/_estimator.py
[Rotbaum]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/rotbaum/_estimator.py
[SAN]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/san/_estimator.py
[TFT]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/tft/_estimator.py
[Transformer]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/transformer/_estimator.py
[WaveNet]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/wavenet/_estimator.py
[SFF_mx]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/simple_feedforward/_estimator.py
[SFF_torch]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/torch/model/simple_feedforward/estimator.py
[DeepVAR]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/deepvar/_estimator.py
[GPVAR]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/gpvar/_estimator.py
[LSTNet]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/lstnet/_estimator.py
[DeepTPP]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/tpp/deeptpp/_estimator.py
[RForecast]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/r_forecast/_predictor.py
[Prophet]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/prophet/_predictor.py
[NaiveSeasonal]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/seasonal_naive/_predictor.py
[Naive2]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/naive_2/_predictor.py
[NPTS]: https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/model/npts/_predictor.py
