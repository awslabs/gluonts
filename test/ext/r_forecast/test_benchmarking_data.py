from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import time
import rpy2
import numpy as np
import boto3
import yaml
import argparse
import pickle

sys.path[0] = '/Users/linmen/Desktop/gluonts/src'
sys.path = [i for i in sys.path if 'autogluon' not in i]

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import Evaluator, backtest_metrics, make_evaluation_predictions
from gluonts.model.predictor import ParallelizedPredictor
import multiprocessing as mp
import matplotlib.pyplot as plt
from gluonts.model import Predictor, QuantileForecast
from gluonts.ext.prophet import ProphetPredictor

from gluonts.ext.r_forecast import (
    RForecastPredictor,
    R_IS_INSTALLED,
    RPY2_IS_INSTALLED,
    UNIVARIATE_QUANTILE_FORECAST_METHODS,
    SUPPORTED_UNIVARIATE_METHODS,
)
from gluonts.time_feature import get_seasonality

fourier_frequency_low_periods = 4
fourier_ratio_threshold_low_periods = 18
fourier_frequency_high_periods = 52
fourier_ratio_threshold_high_periods = 2
fourier_order = 4

def check_and_fill_gaps(dataset, freq):
    max_datetime = dataset.groupby('item_id').apply(lambda x: max(x.timestamp)).unique()
    min_datetime = dataset.groupby('item_id').apply(lambda x: min(x.timestamp)).unique()

    num_of_steps = pd.date_range(min_datetime[0], max_datetime[0], freq = freq)
    num_of_steps_by_item = dataset.groupby('item_id').count().target_value.unique()

    if len(max_datetime) == 1 and len(min_datetime) == 1 and len(num_of_steps_by_item) == 1 and num_of_steps_by_item[0] == len(num_of_steps):
        print('original data does not contain missingness')
        return dataset
    else:
        full_datatime = pd.DataFrame(pd.date_range(start=dataset.timestamp.min(), end=dataset.timestamp.max(), freq = freq), columns=['date'])
        full_datatime.columns = ['timestamp']

        dataset_fill = pd.DataFrame()
        for item in dataset.item_id.unique():
            subset = dataset[dataset.item_id == item]
            diff = pd.DataFrame(set(full_datatime.timestamp) - set(subset.timestamp))
            if len(diff) > 0:
                diff.columns = ['timestamp']
                diff['item_id'] = item
                diff['target_value'] = 0

                subset = pd.concat([subset, diff], axis=0)
            dataset_fill = pd.concat([dataset_fill, subset], axis=0)

        print('missing time index in the original data has been filled')
        return dataset_fill

def preprocess_step(dataset_name):
    with open("../test/ext/r_forecast/datasets.yaml", 'r') as stream:
        data_info = yaml.safe_load(stream)

    freq = data_info[dataset_name]['freq']
    prediction_length = data_info[dataset_name]['prediction_length']
    tts_key = data_info[dataset_name]['tts_key']
    tts_data_columns = data_info[dataset_name]['tts_columns']
    #
    # bucket = data_info[dataset_name]['bucket']
    # s3 = boto3.client('s3')
    # obj = s3.get_object(Bucket=bucket, Key=tts_key)
    # data_tts = pd.read_csv(obj['Body'])
    # data_tts = data_tts[data_tts.item_id == 0]

    data_tts = pd.read_csv(tts_key)
    data_tts.columns = tts_data_columns
    print(f"data dimension is {data_tts.shape}")
    data_tts.timestamp = pd.to_datetime(data_tts.timestamp, format="%Y-%m-%dT%H:%M:%S", errors='coerce')
    data_tts.timestamp = data_tts.timestamp.dt.tz_localize(None)
    feat_dynamic_real = static_feature_columns = []
    data_tts['timestamp'] = data_tts['timestamp'].apply(lambda x: x.replace(minute=0, second=0))
    data_tts = check_and_fill_gaps(data_tts, freq)
    print(data_tts.head())

    data_tts.select_dtypes('number').fillna(0, inplace=True)
    ds = PandasDataset.from_long_dataframe(
        data_tts,
        target="target_value",
        timestamp="timestamp",
        item_id="item_id",
        freq=freq,
        feat_dynamic_real=feat_dynamic_real,
        static_feature_columns=static_feature_columns,
    )

    period = get_seasonality(freq)
    return {"prediction_length": prediction_length,
            "ds": ds,
            "data": data_tts,
            "freq": freq,
            "period": period}

def gluonts_r_runs(method, dataset_name, num_samples = 100):

    training_input = preprocess_step(dataset_name)
    freq = training_input["freq"]
    period = training_input["period"]
    prediction_length = training_input["prediction_length"]
    ds = training_input["ds"]

    start_t = time.time()
    ## Add fourier logic
    raw_data = training_input["data"]
    raw_data = raw_data.set_index('timestamp')

    len_ts = raw_data.groupby('item_id').size().values[0]
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
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    time_elapsed = time.time() - start_t
    agg_metrics['latency'] = time_elapsed

    return agg_metrics, item_metrics, forecasts, raw_data, prediction_length

def make_diagnostic_plot(data, forecasts, prediction_length):

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        choices=["ets", "arima", "prophet"],
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[
            "uber_weather_1item",
            "uber_weather_2items",
            "everlane",
            "Kaggle_retail",
            "uber_weather",
            "electricity",
            "covid_death",
            "solar",
            "osaka_gas",
            "kdd_2018",
        ],
    )
    args = parser.parse_args()
    method = args.method
    dataset_name = args.dataset_name
    num_samples = args.num_samples
    agg_metrics, item_metrics, forecasts, data, prediction_length = gluonts_r_runs(method, dataset_name, num_samples)

    print(f"agg_metrics is {agg_metrics}")
    print(f"item_metrics is {item_metrics}")
    f_name = f"{method}-{dataset_name}-AggMetrics.pkl"
    with open(f_name, "wb") as f:
        pickle.dump(agg_metrics, f)

    make_diagnostic_plot(data, forecasts, prediction_length)
