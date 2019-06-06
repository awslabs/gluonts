# gluon-ts evaluations

This folder aims at collecting evaluations of forecasting models. The goal is to make reproducibility and comparison easier by versioning the code producing dataset as well as the model and evaluation code.


## mean_wQuantileLoss

estimator | electricity | exchange_rate | m4_daily | m4_hourly | m4_monthly | m4_quarterly | m4_weekly | m4_yearly | solar-energy | traffic
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
DeepAREstimator | 0.085 | 0.009 | 0.025 | 0.033 | 0.115 | 0.087 | 0.048 | 0.128 | 0.398 | 0.126
MQCNNEstimator | 0.102 | 0.016 | 0.028 | 0.065 | 0.123 | 0.089 | 0.060 | 0.124 | 0.627 | 0.274
NPTSPredictor | 0.144 | 0.021 | 0.145 | 0.048 | 0.233 | 0.255 | 0.297 | 0.355 | 0.825 | 0.179
RForecastPredictor |  | 0.008 |  |  |  |  |  |  |  | 
SeasonalNaivePredictor | 0.139 | 0.011 | 0.028 | 0.048 | 0.146 | 0.119 | 0.063 | 0.161 | 1.000 | 0.251
SimpleFeedForwardEstimator | 0.095 | 0.009 | 0.023 | 0.044 | 0.119 | 0.085 | 0.053 | 0.127 | 0.436 | 0.213

## ND

estimator | electricity | exchange_rate | m4_daily | m4_hourly | m4_monthly | m4_quarterly | m4_weekly | m4_yearly | solar-energy | traffic
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
DeepAREstimator | 0.100 | 0.011 | 0.030 | 0.042 | 0.125 | 0.102 | 0.060 | 0.152 | 0.490 | 0.150
MQCNNEstimator | 0.118 | 0.018 | 0.034 | 0.085 | 0.131 | 0.104 | 0.068 | 0.150 | 0.707 | 0.311
NPTSPredictor | 0.185 | 0.025 | 0.190 | 0.063 | 0.293 | 0.334 | 0.389 | 0.442 | 1.026 | 0.225
RForecastPredictor |  | 0.009 |  |  |  |  |  |  |  | 
SeasonalNaivePredictor | 0.139 | 0.011 | 0.028 | 0.048 | 0.146 | 0.119 | 0.063 | 0.161 | 1.000 | 0.251
SimpleFeedForwardEstimator | 0.114 | 0.012 | 0.027 | 0.055 | 0.130 | 0.100 | 0.062 | 0.156 | 0.525 | 0.254

## RMSE

estimator | electricity | exchange_rate | m4_daily | m4_hourly | m4_monthly | m4_quarterly | m4_weekly | m4_yearly | solar-energy | traffic
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
DeepAREstimator | 14238.631 | 0.014 | 635.905 | 1344.878 | 1405.169 | 1405.009 | 613.102 | 1865.383 | 31.510 | 0.025
MQCNNEstimator |  |  |  |  |  |  |  |  |  | 
NPTSPredictor | 22466.171 | 0.033 | 2205.998 | 2837.655 | 2614.135 | 3250.483 | 3638.738 | 4213.817 | 53.566 | 0.031
RForecastPredictor |  | 0.011 |  |  |  |  |  |  |  | 
SeasonalNaivePredictor | 17590.598 | 0.013 | 705.425 | 1901.146 | 1628.794 | 1577.303 | 673.443 | 2016.458 | 62.518 | 0.037
SimpleFeedForwardEstimator | 15396.873 | 0.014 | 646.524 | 2360.828 | 1446.087 | 1442.306 | 683.172 | 1972.476 | 35.511 | 0.034


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


