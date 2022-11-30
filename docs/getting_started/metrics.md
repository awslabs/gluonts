## Available metrics

This is a list of metrics provided by the `ev` module. For implementation details see [here](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ev/metrics.py).

| Metric name                 | Default parameters    | Aggregation |
| --------------------------- | --------------------- | ----------- |
| mean_absolute_label         | -                     | Mean        |
| sum_absolute_label          | -                     | Sum         |
| MSE                         | forecast type: mean   | Mean        |
| RMSE                        | forecast type: mean   | Mean        |
| NRMSE                       | forecast type: mean   | Mean        |
| SumError                    | forecast type: median | Sum         |
| SumAbsoluteError            | forecast type: median | Sum         |
| MAPE                        | forecast type: median | Mean        |
| SMAPE                       | forecast type: median | Mean        |
| MASE                        | forecast type: median | Mean        |
| ND                          | forecast type: median | Mean        |
| OWA                         | forecast type: median | Mean        |
| MSIS                        | alpha: 0.05           | Mean        |
| SumQuantileLoss             | -                     | Mean        |
| WeightedSumQuantileLoss     | -                     | Mean        |
| Coverage                    | -                     | Mean        |
| MeanSumQuantileLoss         | -                     | Mean        |
| MeanWeightedSumQuantileLoss | -                     | Mean        |
| MAECoverage                 | -                     | Mean        |
