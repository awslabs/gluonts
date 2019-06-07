# Third-party imports
import numpy as np
import pandas as pd
import pytest
from pydantic import PositiveInt

# First-party imports
from gluonts.dataset.common import Dataset
from gluonts.model.seasonal_naive._predictor import SeasonalNaivePredictor


def generate_random_dataset(
    num_ts: int, start_time: str, freq: str, min_length: int, max_length: int
) -> Dataset:
    start_timestamp = pd.Timestamp(start_time, freq=freq)
    for _ in range(num_ts):
        ts_length = np.random.randint(low=min_length, high=max_length)
        target = np.random.uniform(size=(ts_length,))
        data = {"target": target, "start": start_timestamp}
        yield data


PREDICTION_LENGTH = PositiveInt(30)
SEASON_LENGTH = PositiveInt(210)
START_TIME = "2018-01-03 14:37:12"  # That's a Wednesday
MIN_LENGTH = 300
MAX_LENGTH = 400
NUM_TS = 10


@pytest.mark.parametrize(
    "freq", ["1min", "15min", "30min", "1H", "2H", "12H", "7D", "1W", "1M"]
)
def test_seasonal_naive(freq):
    predictor = SeasonalNaivePredictor(freq, PREDICTION_LENGTH, SEASON_LENGTH)
    dataset = list(
        generate_random_dataset(
            num_ts=NUM_TS,
            start_time=START_TIME,
            freq=freq,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
        )
    )

    # get forecasts
    forecasts = list(predictor.predict(dataset))

    assert len(dataset) == NUM_TS
    assert len(forecasts) == NUM_TS

    # check forecasts are as expected
    for data, forecast in zip(dataset, forecasts):
        assert forecast.samples.shape == (1, PREDICTION_LENGTH)

        ref = data["target"][
            -SEASON_LENGTH : -SEASON_LENGTH + PREDICTION_LENGTH
        ]

        data_start = pd.Timestamp(START_TIME, freq=freq)
        exp_forecast_start = data_start + len(data["target"]) * data_start.freq

        assert forecast.start_date == exp_forecast_start
        assert np.allclose(forecast.samples[0], ref)
