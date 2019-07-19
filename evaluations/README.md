# gluon-ts evaluations

This folder aims at collecting evaluations of forecasting models. The goal is to make reproducibility and comparison easier by versioning the code producing dataset as well as the model and evaluation code. 

Note that the evaluations are not "optimal" in the sense that the models are trained with default parameters for all the datasets, and with no additional features associated to the datasets.

## mean_wQuantileLoss

estimator | electricity | exchange_rate | m4_daily | m4_hourly | m4_monthly | m4_quarterly | m4_weekly | m4_yearly | solar-energy | traffic
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
DeepAREstimator | 0.050 | 0.023 | 0.025 | 0.033 | 0.115 | 0.087 | 0.048 | 0.128 | 0.398 | 0.126
MQCNNEstimator | 0.083 | 0.016 | 0.027 | 0.065 | 0.124 | 0.089 | 0.059 | 0.122 | 0.551 | 0.272
MQRNNEstimator | 0.197 | 0.004 | 0.222 | 0.298 | 0.209 | 0.326 | 0.104 | 0.328 | 0.164 | 0.087
NPTSPredictor | 0.062 | 0.021 | 0.145 | 0.048 | 0.233 | 0.255 | 0.296 | 0.355 | 0.826 | 0.180
RForecastPredictor_arima |  | 0.008 | 0.024 | 0.040 |  | 0.080 | 0.050 | 0.124 | 1.153 |
RForecastPredictor_ets | 0.121 | 0.008 | 0.023 | 0.043 | 0.099 | 0.079 | 0.051 | 0.126 | 1.778 | 0.373
SeasonalNaivePredictor | 0.070 | 0.011 | 0.028 | 0.048 | 0.146 | 0.119 | 0.063 | 0.161 | 1.000 | 0.251
SimpleFeedForwardEstimator | 0.062 | 0.009 | 0.023 | 0.044 | 0.116 | 0.088 | 0.051 | 0.132 | 0.435 | 0.212
TransformerEstimator | 0.066 | 0.009 | 0.027 | 0.035 | 0.136 | 0.105 | 0.083 | 0.160 | 0.432 | 0.132

## ND

estimator | electricity | exchange_rate | m4_daily | m4_hourly | m4_monthly | m4_quarterly | m4_weekly | m4_yearly | solar-energy | traffic
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
DeepAREstimator | 0.061 | 0.029 | 0.030 | 0.042 | 0.125 | 0.102 | 0.060 | 0.152 | 0.490 | 0.150
MQCNNEstimator | 0.102 | 0.019 | 0.032 | 0.086 | 0.132 | 0.103 | 0.065 | 0.146 | 0.666 | 0.310
MQRNNEstimator | 0.639 | 0.015 | 0.662 | 0.906 | 0.660 | 0.981 | 0.345 | 0.987 | 0.702 | 0.334
NPTSPredictor | 0.080 | 0.025 | 0.191 | 0.063 | 0.293 | 0.334 | 0.387 | 0.442 | 1.031 | 0.225
RForecastPredictor_arima |  | 0.009 | 0.029 | 0.053 |  | 0.097 | 0.060 | 0.148 | 1.150 |
RForecastPredictor_ets | 0.150 | 0.010 | 0.027 | 0.054 | 0.120 | 0.095 | 0.061 | 0.149 | 1.364 | 0.385
SeasonalNaivePredictor | 0.070 | 0.011 | 0.028 | 0.048 | 0.146 | 0.119 | 0.063 | 0.161 | 1.000 | 0.251
SimpleFeedForwardEstimator | 0.075 | 0.012 | 0.028 | 0.055 | 0.126 | 0.104 | 0.060 | 0.158 | 0.520 | 0.251
TransformerEstimator | 0.082 | 0.011 | 0.032 | 0.043 | 0.150 | 0.128 | 0.098 | 0.193 | 0.534 | 0.159

## RMSE

estimator | electricity | exchange_rate | m4_daily | m4_hourly | m4_monthly | m4_quarterly | m4_weekly | m4_yearly | solar-energy | traffic
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
DeepAREstimator | 1177.808 | 0.033 | 635.905 | 1344.878 | 1405.169 | 1405.009 | 613.102 | 1865.383 | 31.510 | 0.025
MQCNNEstimator |  |  |  |  |  |  |  |  |  |
MQRNNEstimator |  |  |  |  |  |  |  |  |  |
NPTSPredictor | 1679.833 | 0.033 | 2207.532 | 2871.974 | 2613.715 | 3251.401 | 3621.983 | 4211.343 | 53.450 | 0.031
RForecastPredictor_arima |  | 0.011 | 641.476 | 2285.035 |  | 1436.552 | 644.820 | 2065.602 | 58.934 |
RForecastPredictor_ets | 3195.747 | 0.012 | 602.283 | 2158.406 | 1413.275 | 1374.529 | 659.644 | 2066.347 | 65.986 | 0.039
SeasonalNaivePredictor | 1139.925 | 0.013 | 705.425 | 1901.146 | 1628.794 | 1577.303 | 673.443 | 2016.458 | 62.518 | 0.037
SimpleFeedForwardEstimator | 1285.875 | 0.014 | 677.479 | 2323.024 | 1420.506 | 1453.103 | 672.740 | 1982.234 | 37.251 | 0.034
TransformerEstimator | 2059.355 | 0.014 | 664.324 | 1575.836 | 1512.068 | 1494.884 | 848.616 | 2010.936 | 35.152 | 0.026

# FAQ

## Can I add evaluations of a model? Are there conditions to add evaluations?
The accuracy numbers should be obtained with code checked in gluon-ts to allow other researcher to reproduce reported results. 
This can include models written in python but also wrappers (for instance see `RForecastPredictor` that allows to use Hyndman R forecast package) or not completely polished code (unpolished code will be put under a `contribution` folder). 

Also models are run with default parameters, they should only require as parameters the time frequency of the data and the number of prediction steps needed. All others should be fixed or adapted automatically to the data.

If there is sufficient demand, we could also collect results that are run outside of gluon-ts but one will then not be able to reproduce results.


## How can I add evaluations of my model?
Run `generate_evaluations.py` which will save evaluations results in `evaluations`. The results can then be visualised with `show_results.py` (which generates the table above). 
You can then issue a pull-request with your model, adding or updating evaluations files.


## How do you enforce that each number is valid?
We do not enforce that the results are actually produced by the code (one could for instance put arbitrary 
low numbers). 
However, every result is versioned through git together that the code that produced it and can be checked by anyone 
by rerunning a given evaluation. We might also at some point generate this table automatically.


## Can I add another dataset?
We are happy to include other datasets.
To add another dataset, you have to include the downloading and processing code in 
`gluonts.dataset.repository.datasets.py`.


