import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import Evaluator, backtest_metrics, make_evaluation_predictions
from gluonts.model.predictor import ParallelizedPredictor
import multiprocessing as mp
from gluonts.ext.prophet import ProphetPredictor

from gluonts.ext.r_forecast import RForecastPredictor
from gluonts.time_feature import get_seasonality

def gluonts_r_runs(data, method, prediction_length, freq, num_samples = 100):

    ds = PandasDataset.from_long_dataframe(
        data,
        target="target_value",
        timestamp="timestamp",
        item_id="item_id",
        freq=freq,
    )
    period = get_seasonality(freq)

    if 'arima' in method.lower():
        predictor = RForecastPredictor(
            **{
                "freq": freq,
                "prediction_length": prediction_length,
                "method_name": method,
                "period": period,
                "params": {
                    "output_types": ["samples", "mean", "quantiles"],
                    "quantiles": [0.50, 0.10, 0.90]
                }
            }
        )
        predictor = ParallelizedPredictor(predictor, num_workers=mp.cpu_count())
    elif method == 'ets':
        predictor = RForecastPredictor(
            **{
                "freq": freq,
                "prediction_length": prediction_length,
                "method_name": method,
                "period": period,
                "params": {
                    "output_types": ["samples", "mean", "quantiles"],
                    "quantiles": [0.50, 0.10, 0.90]
                }
            }
        )
        predictor = ParallelizedPredictor(predictor, num_workers=mp.cpu_count())
    elif method == "prophet":
        predictor = ProphetPredictor(
            **{
                "prediction_length": prediction_length,
            }
        )
        predictor = ParallelizedPredictor(predictor, num_workers=mp.cpu_count())
    else:
        raise ValueError(f"{method} unsupported: only ARIMA and ETS from Statsforecast can be evaluated in this benchmark")

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=ds,
        predictor=predictor,
        num_samples=num_samples,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    agg_metrics, _ = evaluator(tss, forecasts)
    return agg_metrics, forecasts

def make_diagnostic_plot(data, forecasts):

    plt.plot(data.target_value.values[-prediction_length:], color = 'green')
    plt.plot(forecasts[0]['0.5'], color = 'darkblue')
    plt.plot(forecasts[0]['mean'], color = 'blue')

    plt.fill_between(
        np.arange(1, prediction_length + 1), forecasts[0]["0.1"], forecasts[0]["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
    )
    plt.legend()
    plt.show()

if __name__ == '__main__':

    ## example dataset in /test/ext/r_forecast directory
    num_samples = 100
    method = 'fourier.arima'
    prediction_length = 168
    freq = 'H'

    test_data = pd.read_csv('/Users/linmen/Desktop/gluonts/test/ext/r_forecast/uber_weather.csv')
    test_data.timestamp = pd.to_datetime(test_data.timestamp, format="%Y-%m-%dT%H:%M:%S", errors='coerce')
    test_data.timestamp = test_data.timestamp.dt.tz_localize(None)
    test_data.select_dtypes('number').fillna(0, inplace=True)

    agg_metrics, forecasts = gluonts_r_runs(test_data, method, prediction_length, freq, num_samples)

    print(f"agg_metrics is {agg_metrics}")
    make_diagnostic_plot(test_data, forecasts)
