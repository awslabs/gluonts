# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np
import pytest
import matplotlib.pyplot as plt 
from gluonts.core import serde
from gluonts.dataset.common import ListDataset
from gluonts.ext.r_forecast import RForecastPredictor
from gluonts.evaluation import Evaluator, backtest_metrics, make_evaluation_predictions

freq = 'H'
period = 24

## two weeks of data
dataset = ListDataset(
    data_iter=[
        {
            "start": "1990-01-01",
            "target": np.array([item for i in range(70) for item in np.sin(2 * np.pi/period * np.arange(1, period + 1, 1))]) 
                + np.random.normal(0, 0.5, period * 70)
                + np.array([item for i in range(10) for item in [0 for i in range(5 * 24)] + [8 for i in range(4)] + [0 for i in range(20)] + [8 for i in range(4)] + [0 for i in range(20)]]), 
        }
    ],
    freq=freq,
)
params = dict(freq=freq, prediction_length=24*7, period=period, params={"quantiles": [0.50, 0.10, 0.90], "output_types": ["mean", "samples", "quantiles"]})

@pytest.mark.parametrize(
    "method",
    [
        'arima', 
        'fourier.arima',
    ],
)
def test_arima(method: str):

    predictor = RForecastPredictor(**params, method_name=method)

    act_fcst = next(predictor.predict(dataset))
    act_fcst = next(predictor.predict(dataset))
    exp_fcst = dataset[0]['target'][:period * 7]

    assert exp_fcst.shape == act_fcst.quantile(0.1).shape
    assert exp_fcst.shape == act_fcst.quantile(0.5).shape
    assert exp_fcst.shape == act_fcst.quantile(0.9).shape

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=100,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    agg_metrics, _ = Evaluator(quantiles=[0.1, 0.5, 0.9])(tss, forecasts)
    assert isinstance(agg_metrics, dict)
    assert "MAPE" in agg_metrics.keys()

def test_compare_arimas():

    def evaluate(predictor):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=100,
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        agg_metrics, _ = Evaluator(quantiles=[0.1, 0.5, 0.9])(tss, forecasts)

        return agg_metrics

    predictor = RForecastPredictor(**params, method_name='arima')
    arima_eval_metrics = evaluate(predictor)

    predictor = RForecastPredictor(**params, method_name='fourier.arima')
    fourier_arima_eval_metrics = evaluate(predictor)

    assert fourier_arima_eval_metrics["MASE"] < arima_eval_metrics["MASE"]





