# GluonTS evaluations

This folder aims at collecting evaluations of forecasting models. The goal is to make reproducibility and comparison easier by versioning the code producing dataset as well as the model and evaluation code. 

Note that the evaluations are not "optimal" in the sense that the models are trained with the same hyper-parameters across all the datasets, and with no additional features associated to the datasets.

## Configuration

## Hyper-parameters:

```
hps = {
  epochs: 300
  num_batches_per_epoch: 100
  batch_size: 32
}
```
(Exception: MQCNNEstimator was trained using `hybridize=False` since `hybridize=True` is buggy with `mxnet-cu92>=1.6.0`)

One can create the corresponding Estimators using the `from_hyperparameters` function, e.g.: `my_deepAREstimator = DeepAREstimator.from_hyperparameters(**hps)`.

## System configuration & Commit:

- gluon-ts commit: `4aec7ee`
- instance type: `ml.p2.xlarge`
- ipdated dockerfiles where used

# Results

## mean_wQuantileLoss


|                      |   ('electricity', 'mean') |   ('electricity', 'std') |   ('exchange_rate', 'mean') |   ('exchange_rate', 'std') |   ('m4_daily', 'mean') |   ('m4_daily', 'std') |   ('m4_hourly', 'mean') |   ('m4_hourly', 'std') |   ('m4_monthly', 'mean') |   ('m4_monthly', 'std') |   ('m4_quarterly', 'mean') |   ('m4_quarterly', 'std') |   ('m4_weekly', 'mean') |   ('m4_weekly', 'std') |   ('m4_yearly', 'mean') |   ('m4_yearly', 'std') |   ('solar-energy', 'mean') |   ('solar-energy', 'std') |   ('traffic', 'mean') |   ('traffic', 'std') |   ('wiki-rolling_nips', 'mean') |   ('wiki-rolling_nips', 'std') |   ('taxi_30min', 'mean') |   ('taxi_30min', 'std') |
|:---------------------|--------------------------:|-------------------------:|----------------------------:|---------------------------:|-----------------------:|----------------------:|------------------------:|-----------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|---------------------------:|--------------------------:|----------------------:|---------------------:|--------------------------------:|-------------------------------:|-------------------------:|------------------------:|
| DeepAREstimator            |                     0.063 |                    0.005 |                       0.01  |                      0.003 |                  0.036 |                 0.005 |                   0.294 |                  0.106 |                    0.123 |                   0.012 |                      0.085 |                     0.002 |                   0.06  |                  0.005 |                   0.129 |                  0.007 |                      0.358 |                     0.003 |                 0.123 |                0.001 |                          15.3   |                          0.175 |                    0.375 |                   0.032 |
| MQCNNEstimator             |                     0.068 |                    0.001 |                       0.013 |                      0     |                  0.029 |                 0.002 |                   0.044 |                  0.002 |                    0.118 |                   0.001 |                      0.09  |                     0.001 |                   0.079 |                  0.004 |                   0.123 |                  0.004 |                      0.768 |                     0.016 |                 1.379 |                0.1   |                           0.236 |                          0.002 |                    0.622 |                   0.077 |
| TransformerEstimator       |                     0.081 |                    0.007 |                       0.009 |                      0.002 |                  0.052 |                 0.012 |                   0.461 |                  0.136 |                    0.126 |                   0.005 |                      0.089 |                     0.003 |                   0.064 |                  0.007 |                   0.152 |                  0.027 |                      0.4   |                     0.01  |                 0.122 |                0.002 |                          28.027 |                          0.17  |                    0.343 |                   0.006 |
| SimpleFeedForwardEstimator |                     0.06  |                    0     |                       0.009 |                      0     |                  0.023 |                 0.001 |                   0.048 |                  0.003 |                    0.116 |                   0.003 |                      0.087 |                     0.002 |                   0.051 |                  0.001 |                   0.123 |                  0.002 |                      0.439 |                     0.006 |                 0.211 |                0.001 |                           0.248 |                          0.003 |                    0.429 |                   0.002 |
| SeasonalNaivePredictor     |                     0.07  |                         |                       0.011 |                           |                  0.028 |                      |                   0.048 |                       |                    0.146 |                        |                      0.119 |                          |                   0.063 |                       |                   0.161 |                       |                      1     |                          |                 0.251 |                     |                           0.404 |                               |                    0.755 |                        |
NPTSPredictor | 0.062 | | 0.021 | | 0.145 | | 0.048 | | 0.233 | | 0.255 | |0.296 | | 0.355 | | 0.826 | | 0.180 | | | | | |
RForecastPredictor_arima |  | | 0.008 | | 0.024 | | 0.040 | |  | | 0.080 | | 0.050 | | 0.124 | | 1.153 | | | | | |
RForecastPredictor_ets | 0.121 | | 0.008 | | 0.023 | | 0.043 | | 0.099 |  |0.079 | | 0.051 | | 0.126 | | 1.778 | | 0.373 | | | | | |

## ND

|                      |   ('electricity', 'mean') |   ('electricity', 'std') |   ('exchange_rate', 'mean') |   ('exchange_rate', 'std') |   ('m4_daily', 'mean') |   ('m4_daily', 'std') |   ('m4_hourly', 'mean') |   ('m4_hourly', 'std') |   ('m4_monthly', 'mean') |   ('m4_monthly', 'std') |   ('m4_quarterly', 'mean') |   ('m4_quarterly', 'std') |   ('m4_weekly', 'mean') |   ('m4_weekly', 'std') |   ('m4_yearly', 'mean') |   ('m4_yearly', 'std') |   ('solar-energy', 'mean') |   ('solar-energy', 'std') |   ('traffic', 'mean') |   ('traffic', 'std') |   ('wiki-rolling_nips', 'mean') |   ('wiki-rolling_nips', 'std') |   ('taxi_30min', 'mean') |   ('taxi_30min', 'std') |
|:---------------------|--------------------------:|-------------------------:|----------------------------:|---------------------------:|-----------------------:|----------------------:|------------------------:|-----------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|---------------------------:|--------------------------:|----------------------:|---------------------:|--------------------------------:|-------------------------------:|-------------------------:|------------------------:|
| DeepAREstimator            |                     0.078 |                    0.006 |                       0.013 |                      0.003 |                  0.043 |                 0.005 |                   0.339 |                  0.093 |                    0.136 |                   0.012 |                      0.103 |                     0.002 |                   0.073 |                  0.004 |                   0.151 |                  0.008 |                      0.473 |                     0.006 |                 0.147 |                0.001 |                           5.496 |                          0.06  |                    0.469 |                   0.042 |
| MQCNNEstimator             |                     0.084 |                    0.001 |                       0.014 |                      0     |                  0.037 |                 0.003 |                   0.05  |                  0.004 |                    0.129 |                   0.001 |                      0.108 |                     0.001 |                   0.092 |                  0.003 |                   0.143 |                  0.005 |                      0.8   |                     0.016 |                 1.481 |                0.052 |                           0.272 |                          0.004 |                    0.746 |                   0.09  |
| TransformerEstimator       |                     0.094 |                    0.007 |                       0.012 |                      0.002 |                  0.064 |                 0.015 |                   0.569 |                  0.205 |                    0.143 |                   0.009 |                      0.105 |                     0.005 |                   0.077 |                  0.009 |                   0.183 |                  0.031 |                      0.497 |                     0.008 |                 0.145 |                0.002 |                          10.041 |                          0.062 |                    0.425 |                   0.009 |
| SimpleFeedForwardEstimator |                     0.073 |                    0     |                       0.011 |                      0     |                  0.028 |                 0.001 |                   0.059 |                  0.003 |                    0.131 |                   0.004 |                      0.104 |                     0.004 |                   0.06  |                  0.001 |                   0.149 |                  0.003 |                      0.528 |                     0.011 |                 0.249 |                0.001 |                           0.294 |                          0.004 |                    0.536 |                   0.002 |
| SeasonalNaivePredictor     |                     0.07  |                         |                       0.011 |                           |                  0.028 |                      |                   0.048 |                       |                    0.146 |                        |                      0.119 |                          |                   0.063 |                       |                   0.161 |                       |                      1     |                          |                 0.251 |                     |                           0.404 |                               |                    0.755 |                        |
NPTSPredictor | 0.080 | | 0.025 | | 0.191 | | 0.063 | | 0.293 | | 0.334 | | 0.387 | | 0.442 | | 1.031 | | 0.225 | | | | | |
RForecastPredictor_arima | | | 0.009 | | 0.029 | | 0.053 | |  | | 0.097 | | 0.060 | | 0.148 | | 1.150 | | | | | |
RForecastPredictor_ets | 0.150 | | 0.010 | |0.027 | |0.054 | | 0.120 | |0.095 | |0.061 | | 0.149 | | 1.364 | | 0.385 | | | | | |

## RMSE

|                      |   ('electricity', 'mean') |   ('electricity', 'std') |   ('exchange_rate', 'mean') |   ('exchange_rate', 'std') |   ('m4_daily', 'mean') |   ('m4_daily', 'std') |   ('m4_hourly', 'mean') |   ('m4_hourly', 'std') |   ('m4_monthly', 'mean') |   ('m4_monthly', 'std') |   ('m4_quarterly', 'mean') |   ('m4_quarterly', 'std') |   ('m4_weekly', 'mean') |   ('m4_weekly', 'std') |   ('m4_yearly', 'mean') |   ('m4_yearly', 'std') |   ('solar-energy', 'mean') |   ('solar-energy', 'std') |   ('traffic', 'mean') |   ('traffic', 'std') |   ('wiki-rolling_nips', 'mean') |   ('wiki-rolling_nips', 'std') |   ('taxi_30min', 'mean') |   ('taxi_30min', 'std') |
|:---------------------|--------------------------:|-------------------------:|----------------------------:|---------------------------:|-----------------------:|----------------------:|------------------------:|-----------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|---------------------------:|--------------------------:|----------------------:|---------------------:|--------------------------------:|-------------------------------:|-------------------------:|------------------------:|
| DeepAREstimator            |                   2030.54 |                  273.634 |                       0.015 |                      0.003 |                708.483 |                31.911 |                16509.3  |               4282.99  |                  1451.62 |                  50.075 |                    1415.27 |                    15.494 |                 747.66  |                 56.996 |                 1907.64 |                 61.377 |                     29.731 |                     0.264 |                 0.024 |                0     |                        90177.1  |                       2535.62  |                    6.181 |                   0.735 |
| MQCNNEstimator             |                   1371.65 |                   61.842 |                       0.016 |                      0     |                695.066 |                 8.011 |                 1843.05 |                193.475 |                  1439.9  |                   5.351 |                    1427.25 |                     4.449 |                 950.231 |                 38.694 |                 1832.83 |                 16.862 |                     48.894 |                     0.884 |                 0.097 |                0.004 |                         6253.3  |                         16.194 |                    8.905 |                   1.069 |
| TransformerEstimator       |                   2223.53 |                  151.055 |                       0.014 |                      0.002 |                853.776 |                98.41  |                26153.8  |               7718.43  |                  1491.13 |                  23.474 |                    1440.57 |                    23.322 |                 763.638 |                 65.267 |                 1974.44 |                155.381 |                     32.876 |                     0.54  |                 0.024 |                0     |                       162509    |                       1294.95  |                    5.327 |                   0.162 |
| SimpleFeedForwardEstimator |                   1233.81 |                   23.554 |                       0.013 |                      0.001 |                680.337 |                 7.925 |                 2535.78 |                133.479 |                  1441.65 |                  11.403 |                    1456.73 |                    18.181 |                 679.319 |                  6.547 |                 1974.79 |                 19.796 |                     36.14  |                     0.749 |                 0.034 |                0     |                         7320.78 |                         55.688 |                    6.763 |                   0.031 |
| SeasonalNaivePredictor     |                   1139.92 |                         |                       0.013 |                           |                705.425 |                      |                 1901.15 |                      |                  1628.79 |                        |                    1577.3  |                          |                 673.443 |                       |                 2016.46 |                       |                     62.518 |                          |                 0.037 |                     |                         8833.61 |                              |                    9.213 |                        |
NPTSPredictor | 1679.833 | | 0.033 | | 2207.532 | | 2871.974 | | 2613.715 | | 3251.401 | | 3621.983 | | 4211.343 | | 53.450 | | 0.031 | | | | | |
RForecastPredictor_arima |  | | 0.011 | | 641.476 | | 2285.035 | |  | | 1436.552 | | 644.820 | | 2065.602 | | 58.934  | | | | | |
RForecastPredictor_ets | 3195.747 | | 0.012 | | 602.283 | | 2158.406 | | 1413.275 | | 1374.529 | | 659.644 | | 2066.347 | |65.986 | | 0.039  | | | | | |

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


