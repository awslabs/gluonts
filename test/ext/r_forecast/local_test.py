import pandas as pd
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path[0] = '/Users/linmen/Desktop/gluonts/src'
sys.path = [i for i in sys.path if 'autogluon' not in i]

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import Evaluator, backtest_metrics, make_evaluation_predictions
from gluonts.model.predictor import ParallelizedPredictor
import multiprocessing as mp
from gluonts.ext.prophet import ProphetPredictor

from gluonts.ext.r_forecast import RForecastPredictor
from gluonts.time_feature import get_seasonality

fourier_frequency_low_periods = 4
fourier_ratio_threshold_low_periods = 18
fourier_frequency_high_periods = 52
fourier_ratio_threshold_high_periods = 2
fourier_order = 4

def gluonts_r_runs(data, method, prediction_length, freq, num_samples = 100):

    ds = PandasDataset.from_long_dataframe(
        data,
        target="target_value",
        timestamp="timestamp",
        item_id="item_id",
        freq=freq,
    )
    period = get_seasonality(freq)

    ## Add fourier logic
    data = data.set_index('timestamp')
    len_ts = data.groupby('item_id').size().values[0]

    fourier_ratio = len_ts / period
    if ((period > fourier_frequency_low_periods and fourier_ratio > fourier_ratio_threshold_low_periods) or
            (period >= fourier_frequency_high_periods and fourier_ratio > fourier_ratio_threshold_high_periods)):
        K = min(fourier_order, np.floor(period / 2))
        xreg_full = []
        for k in range(1, K + 1):
            xreg_full.append(np.sin(2 * np.pi * k / period * np.arange(1, len_ts + 1)))
            xreg_full.append(np.cos(2 * np.pi * k / period * np.arange(1, len_ts + 1)))
        xreg_full = np.transpose(np.array(xreg_full, dtype=np.float32))
        xreg = xreg_full[:-prediction_length, :]
        xreg_future = xreg_full[-prediction_length:, :]
        seasonal = False
    else:
        seasonal = True
        xreg = xreg_future = None

    if method.lower() == 'arima':
        if xreg is not None:
            predictor = RForecastPredictor(
                **{
                    "freq": freq,
                    "prediction_length": prediction_length,
                    "method_name": method,
                    "period": period,
                    "params": {
                        "seasonal": seasonal,
                        "xreg": xreg,
                        "xreg_future": xreg_future,
                        "output_types": ["samples", "mean", "quantiles"],
                        "quantiles": [0.50, 0.10, 0.90]
                    }
                }
            )
        else:
            predictor = RForecastPredictor(
                **{
                    "freq": freq,
                    "prediction_length": prediction_length,
                    "method_name": method,
                    "period": period,
                    "params": {
                        "seasonal": seasonal,
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
    plt.title('Seasonal ARIMA')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    ## example dataset in /test/ext/r_forecast directory
    num_samples = 100
    method = 'prophet'
    prediction_length = 168
    freq = 'H'

    test_data = pd.read_csv('/Users/linmen/Desktop/gluonts/test/ext/r_forecast/uber_weather.csv')
    test_data.timestamp = pd.to_datetime(test_data.timestamp, format="%Y-%m-%dT%H:%M:%S", errors='coerce')
    test_data.timestamp = test_data.timestamp.dt.tz_localize(None)
    test_data.select_dtypes('number').fillna(0, inplace=True)

    agg_metrics, forecasts = gluonts_r_runs(test_data, method, prediction_length, freq, num_samples)

    print(f"agg_metrics is {agg_metrics}")
    make_diagnostic_plot(test_data, forecasts)
