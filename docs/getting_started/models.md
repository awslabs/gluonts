## Available models

Model + Paper                                                | Local/global | Data layout              | Architecture/method | Implementation
-------------------------------------------------------------|--------------|--------------------------|---------------------|----------------
DeepAR<br>[Salinas et al. 2020][Salinas2020]                 | Global       | Univariate               | RNN | [MXNet][DeepAR_mx], [PyTorch][DeepAR_torch]
DeepState<br>[Rangapuram et al. 2018][Rangapuram2018]        | Global       | Univariate               | RNN, state-space model | [MXNet][DeepState]
DeepFactor<br>[Wang et al. 2019][Wang2019]                   | Global       | Univariate               | RNN, state-space model, Gaussian process | [MXNet][DeepFactor]
Deep Renewal Processes<br>[Türkmen et al. 2021][Turkmen2021] | Global       | Univariate               | RNN | [MXNet][DeepRenewal]
GPForecaster                                                 | Global       | Univariate               | MLP, Gaussian process | [MXNet][GP]
MQ-CNN<br>[Wen et al. 2017][Wen2017]                         | Global       | Univariate               | CNN encoder, MLP decoder | [MXNet][MQDNN]
MQ-RNN<br>[Wen et al. 2017][Wen2017]                         | Global       | Univariate               | RNN encoder, MLP encoder | [MXNet][MQDNN]
N-BEATS<br>[Oreshkin et al. 2019][Oreshkin2019]              | Global       | Univariate               | MLP, residual links | [MXNet][NBeats]
Rotbaum<br>[Hasson et al. 2021][Hasson2021]                  | Global       | Univariate               | XGBoost, Quantile Regression Forests, LightGBM, Level Set Forecaster | [Numpy][Rotbaum]
Temporal Fusion Transformer<br>[Lim et al. 2021][Lim2021]    | Global       | Univariate               | LSTM, self attention | [MXNet][TFT_mx], [PyTorch][TFT_torch] 
Transformer<br>[Vaswani et al. 2017][Vaswani2017]            | Global       | Univariate               | MLP, multi-head attention | [MXNet][Transformer]
WaveNet<br>[van den Oord et al. 2016][vanDenOord2016]        | Global       | Univariate               | Dilated convolution | [MXNet][WaveNet]
SimpleFeedForward                                            | Global       | Univariate               | MLP | [MXNet][SFF_mx], [PyTorch][SFF_torch]
DeepNPTS                                                     | Global       | Univariate               | MLP | [PyTorch][DeepNPTS_torch]
MQF2<br>[Kan et al. 2022][Kan2022]                           | Global       | Univariate               | RNN, ICNN | [PyTorch][MQF2_torch]
DeepVAR<br>[Salinas et al. 2019][Salinas2019]                | Global       | Multivariate             | RNN | [MXNet][DeepVAR]
GPVAR<br>[Salinas et al. 2019][Salinas2019]                  | Global       | Multivariate             | RNN, Gaussian process | [MXNet][GPVAR]
LSTNet<br>[Lai et al. 2018][Lai2018]                         | Global       | Multivariate             | LSTM | [MXNet][LSTNet]
DeepTPP<br>[Shchur et al. 2020][Shchur2020]                  | Global       | Multivariate events      | RNN, temporal point process | [MXNet][DeepTPP]
DeepVARHierarchical<br>[Rangapuram et al. 2021][Rangapuram2021]                  | Global       | Hierarchical             | RNN | [MXNet][DeepVARHierarchical]
RForecast<br>[Hyndman et al. 2008][Hyndman2008]              | Local        | Univariate               | ARIMA, ETS, Croston, TBATS | [Wrapped R package][RForecast]
Prophet<br>[Taylor et al. 2017][Taylor2017]                  | Local        | Univariate               | - | [Wrapped Python package][Prophet]
NaiveSeasonal<br>[Hyndman et al. 2018][Hyndman2018]          | Local        | Univariate               | - | [Numpy][NaiveSeasonal]
Naive2<br>[Makridakis et al. 1998][Makridakis1998]           | Local        | Univariate               | - | [Numpy][Naive2]
NPTS                                                         | Local        | Univariate               | - | [Numpy][NPTS]

<!-- Links to bibliography -->

[Rangapuram2021]: https://proceedings.mlr.press/v139/rangapuram21a.html
[Salinas2020]: https://doi.org/10.1016/j.ijforecast.2019.07.001
[Rangapuram2018]: https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
[Wang2019]: https://proceedings.mlr.press/v97/wang19k.html
[Turkmen2021]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764
[Wen2017]: https://arxiv.org/abs/1711.11053
[Oreshkin2019]: https://openreview.net/forum?id=r1ecqn4YwB
[Hasson2021]: https://openreview.net/forum?id=VD3TMzyxKK
[Lim2021]: https://doi.org/10.1016/j.ijforecast.2021.03.012
[Vaswani2017]: https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
[vanDenOord2016]: https://arxiv.org/abs/1609.03499
[Kan2022]: https://arxiv.org/abs/2202.11316
[Salinas2019]: https://proceedings.neurips.cc/paper/2019/hash/0b105cf1504c4e241fcc6d519ea962fb-Abstract.html
[Lai2018]: https://doi.org/10.1145/3209978.3210006
[Shchur2020]: https://arxiv.org/pdf/1909.12127
[Hyndman2008]: https://www.jstatsoft.org/article/view/v027i03
[Taylor2017]: https://doi.org/10.1080/00031305.2017.1380080
[Hyndman2018]: https://otexts.com/fpp2/simple-methods.html#seasonal-na%C3%AFve-method
[Makridakis1998]: https://www.wiley.com/en-ie/Forecasting:+Methods+and+Applications,+3rd+Edition-p-9780471532330

<!-- Links to code -->

[DeepAR_mx]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deepar/_estimator.py
[DeepAR_torch]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/deepar/estimator.py
[DeepState]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deepstate/_estimator.py
[DeepFactor]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deep_factor/_estimator.py
[DeepRenewal]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/renewal/_estimator.py
[GP]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/gp_forecaster/_estimator.py
[MQDNN]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/seq2seq/_mq_dnn_estimator.py
[NBeats]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/n_beats/_estimator.py
[Rotbaum]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ext/rotbaum/_estimator.py
[TFT_mx]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/tft/_estimator.py
[TFT_torch]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/tft/estimator.py
[Transformer]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/transformer/_estimator.py
[WaveNet]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/wavenet/_estimator.py
[SFF_mx]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/simple_feedforward/_estimator.py
[SFF_torch]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/simple_feedforward/estimator.py
[DeepNPTS_torch]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/deep_npts/_estimator.py
[MQF2_torch]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/mqf2/estimator.py
[DeepVAR]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deepvar/_estimator.py
[DeepVARHierarchical]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deepvar_hierarchical/_estimator.py
[GPVAR]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/gpvar/_estimator.py
[LSTNet]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/lstnet/_estimator.py
[DeepTPP]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/tpp/deeptpp/_estimator.py
[RForecast]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ext/r_forecast/_predictor.py
[Prophet]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ext/prophet/_predictor.py
[NaiveSeasonal]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/model/seasonal_naive/_predictor.py
[Naive2]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ext/naive_2/_predictor.py
[NPTS]: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/model/npts/_predictor.py
